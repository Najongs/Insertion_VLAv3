"""
Sensor Encoder Pre-training using CLIP and Gated Auxiliary Learning

This script pre-trains the UnifiedGatedSensorEncoder using a dual-objective strategy:
1.  **CLIP Contrastive Loss**: Aligns the sensor embedding with a fused vision-text
    embedding from a low-dimensional VLM cache. This teaches the encoder to understand
    the overall context of the robot's interaction.
2.  **Auxiliary Gate Loss**: Directly trains the encoder's internal validity gate
    using time-based pseudo-labels. It learns to predict contact likelihood based
    on the temporal position within an episode (e.g., samples from the last 20%
    are considered "contact likely").

This combined approach forces the encoder to learn both high-level semantic context
and low-level physical events, leading to a more robust representation.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import math
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import time
import wandb

# Add project root to import custom modules
import sys
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.Encoder_model import UnifiedGatedSensorEncoder
from vla_datasets.unified_dataset import create_unified_dataloader
from vla_cache_manager import VLACacheManager

# =====================================
# Attention Pooling Module (from cache_clip_vlm_features.py)
# =====================================

class AttentionPooler(nn.Module):
    """
    Pools a sequence of image tokens into a fixed-size embedding using multi-head attention.
    Identical to the implementation in cache_clip_vlm_features.py.
    """
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, output_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            kdim=input_dim,
            vdim=input_dim,
            batch_first=True
        )
        self.norm = nn.LayerNorm(output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Linear(output_dim * 4, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Image tokens of shape (B, N_tokens, D_in)
        Returns:
            torch.Tensor: Pooled embedding of shape (B, D_out)
        """
        B = x.shape[0]
        query = self.query.expand(B, -1, -1)

        attn_output, _ = self.attention(query, x, x)
        attn_output = self.norm(attn_output)

        mlp_output = self.mlp(attn_output)
        return mlp_output.squeeze(1)

# =====================================
# 0. Collate Function for Variable-Length Vision Features
# =====================================

def clip_collate_fn(batch):
    """
    Custom collate function to handle both embedded and raw features.
    - Embedded: [512] -> stack directly
    - Raw: [N_tokens, D_vlm] -> pad to max length then stack
    """
    sensor_data = torch.stack([item["sensor_data"] for item in batch])
    contact_labels = torch.tensor([item["contact_label"] for item in batch])
    is_embedded_flags = torch.tensor([item["is_embedded"] for item in batch])

    # Check if all items in batch have same embedding status
    all_embedded = is_embedded_flags.all().item()
    all_raw = (~is_embedded_flags).all().item()

    if all_embedded:
        # All already embedded: simple stack
        vision_features = torch.stack([item["vision_features"] for item in batch])
        text_features = torch.stack([item["text_features"] for item in batch])
    elif all_raw:
        # All raw features: pad and stack
        text_features = torch.stack([item["text_features"] for item in batch])

        # Find max sequence length and dimension
        max_len = max(item["vision_features"].shape[0] for item in batch)
        vision_dim = batch[0]["vision_features"].shape[-1]

        # Pad vision features
        vision_features_padded = []
        for item in batch:
            vision_feat = item["vision_features"]
            seq_len = vision_feat.shape[0]
            if seq_len < max_len:
                padding = torch.zeros(max_len - seq_len, vision_dim, dtype=vision_feat.dtype)
                vision_feat = torch.cat([vision_feat, padding], dim=0)
            vision_features_padded.append(vision_feat)

        vision_features = torch.stack(vision_features_padded)
    else:
        # Mixed batch: handle separately (shouldn't happen often with good batching)
        raise ValueError(
            f"Mixed batch detected: {is_embedded_flags.sum().item()} embedded, "
            f"{(~is_embedded_flags).sum().item()} raw. Consider using batch_sampler."
        )

    return {
        "sensor_data": sensor_data,
        "vision_features": vision_features,
        "text_features": text_features,
        "is_embedded": all_embedded,
        "contact_label": contact_labels
    }

# =====================================
# 1. Dataset with Time-based Pseudo-Labeling
# =====================================

class SensorCLIPDataset(Dataset):
    def __init__(self, unified_dataset, clip_cache_root: str, contact_threshold: float = 0.8):
        self.unified_dataset = unified_dataset
        self.clip_cache_manager = VLACacheManager(cache_dir=clip_cache_root)
        self.contact_threshold = contact_threshold

        self.samples = []
        self.num_positives = 0
        self.num_negatives = 0
        self._prepare_samples()

    def _prepare_samples(self):
        is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        pbar = tqdm(self.unified_dataset.datasets, desc="Preparing Samples", disable=not is_main_process)
        
        global_idx_offset = 0
        for sub_dataset in pbar:
            episode_len = len(sub_dataset)
            if episode_len == 0:
                continue
            
            for local_idx in range(episode_len):
                global_idx = global_idx_offset + local_idx
                relative_pos = local_idx / (episode_len - 1) if episode_len > 1 else 1.0
                
                is_contact = relative_pos >= self.contact_threshold
                contact_label = 0.9 if is_contact else 0.1

                if is_contact:
                    self.num_positives += 1
                else:
                    self.num_negatives += 1

                self.samples.append({
                    "global_idx": global_idx,
                    "relative_pos": relative_pos,
                    "contact_label": contact_label
                })
            global_idx_offset += episode_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]

        try:
            data = self.unified_dataset[sample_info["global_idx"]]

            # Load raw features from cache
            episode_id = data["episode_id"]
            vlm_idx = data["vlm_idx"]
            prompt_hash = data.get("prompt_hash", "low_dim_cache")

            cached_data = self.clip_cache_manager.load_cache(
                episode_id, vlm_idx, prompt_hash
            )

            if cached_data is None:
                # If cache not found, try with first sample
                if idx != 0:
                    return self.__getitem__(0)
                else:
                    raise ValueError(f"Cache not found for {episode_id}_vlm{vlm_idx} with hash {prompt_hash}")

            # Unpack cached features
            vision_features, text_features = cached_data

            # Check if already embedded (512d) or raw (2048d/3584d)
            if vision_features.dim() == 1 and vision_features.shape[0] == 512:
                # Already embedded: just use as is
                is_embedded = True
                vision_emb = vision_features  # [512]
                text_emb = text_features      # [512]
            else:
                # Raw features: need to embed
                is_embedded = False

                # Normalize vision features to always be 2D [N_tokens, D]
                if vision_features.dim() == 3:
                    # [1, N, D] -> [N, D]
                    vision_features = vision_features.squeeze(0)
                elif vision_features.dim() == 1:
                    # [D] -> [1, D] (single token)
                    vision_features = vision_features.unsqueeze(0)
                # If already 2D [N, D], keep as is

                # Normalize text features to always be 1D [D]
                if text_features.dim() == 2:
                    # [1, D] -> [D]
                    text_features = text_features.squeeze(0)
                elif text_features.dim() == 0:
                    # Scalar -> [1] (shouldn't happen, but defensive)
                    text_features = text_features.unsqueeze(0)
                # If already 1D [D], keep as is

                vision_emb = vision_features  # [N_tokens, D_vlm]
                text_emb = text_features      # [D_vlm]

            return {
                "sensor_data": data["sensor_data"],
                "vision_features": vision_emb,
                "text_features": text_emb,
                "is_embedded": is_embedded,
                "contact_label": sample_info["contact_label"]
            }
        except Exception as e:
            # On error, return the first valid sample
            if idx != 0:
                return self.__getitem__(0)
            else:
                raise e

# =====================================
# 2. CLIP Model Definition
# =====================================

class CLIPModel(nn.Module):
    def __init__(self, sensor_encoder: UnifiedGatedSensorEncoder, embedding_dim: int = 512, nhead: int = 8):
        super().__init__()
        self.sensor_encoder = sensor_encoder
        self.embedding_dim = embedding_dim

        # Feature compression modules for raw features
        # Support both 3B (2048d) and 7B (3584d) models
        self.vision_pooler_2048 = AttentionPooler(2048, embedding_dim, num_heads=nhead)
        self.vision_pooler_3584 = AttentionPooler(3584, embedding_dim, num_heads=nhead)
        self.text_projector_2048 = nn.Linear(2048, embedding_dim)
        self.text_projector_3584 = nn.Linear(3584, embedding_dim)

        # VLM fusion module
        self.vlm_fusion = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=nhead,
            batch_first=True
        )
        self.vlm_norm = nn.LayerNorm(embedding_dim)

    def forward(self, sensor_data, vision_features, text_features, is_embedded):
        # Sensor encoding
        sensor_embedding, gate_logit = self.sensor_encoder(sensor_data)

        if is_embedded:
            # Already embedded: use directly
            # vision_features: [B, 512], text_features: [B, 512]
            vision_embedding = vision_features
            text_embedding = text_features
        else:
            # Raw features: compress based on dimension
            # Detect VLM dimension from text features
            vlm_dim = text_features.shape[-1]

            if vlm_dim == 2048:
                # 3B model
                vision_embedding = self.vision_pooler_2048(vision_features)  # [B, N, 2048] -> [B, 512]
                text_embedding = self.text_projector_2048(text_features)     # [B, 2048] -> [B, 512]
            elif vlm_dim == 3584:
                # 7B model
                vision_embedding = self.vision_pooler_3584(vision_features)  # [B, N, 3584] -> [B, 512]
                text_embedding = self.text_projector_3584(text_features)     # [B, 3584] -> [B, 512]
            else:
                raise ValueError(f"Unsupported VLM dimension: {vlm_dim}. Expected 2048 or 3584")

        # VLM feature fusion (text queries vision)
        fused_vlm, _ = self.vlm_fusion(
            query=text_embedding.unsqueeze(1),
            key=vision_embedding.unsqueeze(1),
            value=vision_embedding.unsqueeze(1)
        )
        fused_vlm = self.vlm_norm(fused_vlm.squeeze(1))

        # Normalize embeddings for contrastive loss
        sensor_embedding = F.normalize(sensor_embedding, dim=-1)
        fused_vlm = F.normalize(fused_vlm, dim=-1)

        return sensor_embedding, fused_vlm, gate_logit


# =====================================
# 2.5 Sensor Data Augmentation
# =====================================

class SensorNoiseAugmentation(Dataset):
    """
    A wrapper dataset that applies Gaussian noise to the sensor_data tensor.
    This should only be used for the training dataset.
    """
    def __init__(self, subset, noise_std: float = 0.01):
        self.subset = subset
        self.noise_std = noise_std
        if self.noise_std <= 0:
            # Using print directly as this is inside a DDP-spawned process
            # where logging might not be configured yet.
            import os
            if os.environ.get("LOCAL_RANK", "0") == "0":
                print(f"⚠️  SensorNoiseAugmentation created with noise_std <= 0. No noise will be added.")

    def __getitem__(self, idx):
        data = self.subset[idx]
        if self.noise_std > 0:
            # The collate_fn handles tensor conversion, but we ensure it's a tensor for noise addition
            sensor_tensor = data["sensor_data"]
            if not isinstance(sensor_tensor, torch.Tensor):
                sensor_tensor = torch.from_numpy(sensor_tensor)

            noise = torch.randn_like(sensor_tensor) * self.noise_std
            data["sensor_data"] = sensor_tensor + noise
        return data

    def __len__(self):
        return len(self.subset)

# =====================================
# 3. Loss Functions
# =====================================

def siglip_loss(a: torch.Tensor, b: torch.Tensor, scale):
    logits = scale * (a @ b.T)
    labels = torch.eye(logits.size(0), device=logits.device, dtype=logits.dtype)
    loss_a = F.binary_cross_entropy_with_logits(logits, labels)
    loss_b = F.binary_cross_entropy_with_logits(logits.T, labels)
    return 0.5 * (loss_a + loss_b)

# =====================================
# 4. Main Training Function
# =====================================

def main(args):
    # DDP Setup
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    is_main_process = local_rank == 0

    # Model Setup
    sensor_encoder = UnifiedGatedSensorEncoder(output_dim=args.embedding_dim)
    model = CLIPModel(
        sensor_encoder,
        embedding_dim=args.embedding_dim,
    ).to(device)

    # Load pre-trained pooler and projector weights if available (for 7B model)
    pooler_path = Path(args.checkpoint_dir) / "vision_pooler.pth"
    projector_path = Path(args.checkpoint_dir) / "text_projector.pth"

    if pooler_path.exists() and projector_path.exists():
        if is_main_process:
            print(f"Loading pre-trained 7B pooler/projector from {args.checkpoint_dir}")
        try:
            model.vision_pooler_3584.load_state_dict(torch.load(pooler_path, map_location=device))
            model.text_projector_3584.load_state_dict(torch.load(projector_path, map_location=device))
            if is_main_process:
                print("✓ Successfully loaded pre-trained 7B weights")
        except Exception as e:
            if is_main_process:
                print(f"⚠️ Failed to load weights: {e}")
                print("   Starting with randomly initialized pooler and projector")
    else:
        if is_main_process:
            print(f"⚠️ Pre-trained weights not found at {args.checkpoint_dir}")
            print("   Starting with randomly initialized poolers and projectors")

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # logit_scale is the learnable temperature parameter for the contrastive loss
    logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), device=device))

    # Initialize optimizer with model parameters first, then add logit_scale as a separate group.
    # This is a workaround for a potential torch.compile issue with mixed parameter lists.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    optimizer.add_param_group({'params': [logit_scale], 'lr': args.learning_rate, 'weight_decay': 0.01})
    
    # Dataset
    unified_dataset = create_unified_dataloader(
        new_dataset_paths=args.new_dataset_paths,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        return_dataset=True,
        use_cache=False,
        skip_dataset_stats=True,
    )
    
    full_dataset = SensorCLIPDataset(
        unified_dataset,
        clip_cache_root=Path(args.cache_root) / "clip_vlm_features",
        contact_threshold=args.contact_threshold
    )

    # Calculate pos_weight for BCEWithLogitsLoss to handle class imbalance
    if is_main_process:
        print(f"✓ Dataset prepared: {len(full_dataset)} samples")
        print(f"  - Positive (contact) samples: {full_dataset.num_positives}")
        print(f"  - Negative (no-contact) samples: {full_dataset.num_negatives}")

    if full_dataset.num_positives > 0:
        pos_weight_value = full_dataset.num_negatives / full_dataset.num_positives
    else:
        pos_weight_value = 1.0  # Avoid division by zero, default to no weight

    pos_weight_tensor = torch.tensor([pos_weight_value], device=device)
    
    if is_main_process:
        print(f"✓ Calculated pos_weight for gate loss: {pos_weight_value:.2f}")

    # Split and create DataLoaders
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Apply noise augmentation only to the training set
    train_augmented_dataset = SensorNoiseAugmentation(train_subset, noise_std=args.sensor_noise_std)
    if is_main_process:
        print(f"✓ Applying sensor noise augmentation to training set with std: {args.sensor_noise_std}")

    train_sampler = DistributedSampler(train_augmented_dataset, shuffle=True)
    train_loader = DataLoader(
        train_augmented_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=clip_collate_fn
    )

    val_sampler = DistributedSampler(val_subset, shuffle=False)
    val_loader = DataLoader(
        val_subset, # Use the original validation subset
        batch_size=args.batch_size * 2,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=clip_collate_fn
    )

    # Scheduler and Loss
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * args.epochs, eta_min=1e-7)
    gate_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')

    if args.resume_from and Path(args.resume_from).exists():
        if is_main_process:
            print(f"Loading weights from checkpoint: {args.resume_from}")
        
        checkpoint = torch.load(args.resume_from, map_location=device)
        
        # Load model weights and logit_scale, but re-initialize optimizer and scheduler
        model.module.load_state_dict(checkpoint['model_state_dict'])
        
        if 'logit_scale' in checkpoint:
            logit_scale.data = checkpoint['logit_scale']
        
        if is_main_process:
            print(f"✓ Loaded model weights from {args.resume_from}. Starting training from scratch.")
        
        # Note: start_epoch, global_step, and best_val_loss are NOT restored to start fresh.
    else:
        if args.resume_from:
            if is_main_process:
                print(f"⚠️ --resume_from was set, but checkpoint not found at {args.resume_from}. Starting from scratch.")
        os.makedirs(args.checkpoint_dir, exist_ok=True)


    # Initialize wandb (only on main process)
    if is_main_process:
        wandb.init(
            project="QwenVLA-SensorCLIP",
            name=f"sensor_clip_{time.strftime('%Y%m%d_%H%M')}",
            resume="allow",
            id=f"sensor_clip_{int(time.time())}",
            settings=wandb.Settings(start_method="thread", _disable_stats=True),
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "embedding_dim": args.embedding_dim,
                "gate_loss_weight": args.gate_loss_weight,
                "contact_threshold": args.contact_threshold,
            }
        )

    # Training Loop
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not is_main_process)

        for batch in pbar:
            sensor_data = batch["sensor_data"].to(device, dtype=torch.bfloat16)
            vision_features = batch["vision_features"].to(device, dtype=torch.bfloat16)
            text_features = batch["text_features"].to(device, dtype=torch.bfloat16)
            is_embedded = batch["is_embedded"]
            contact_labels = batch["contact_label"].to(device, dtype=torch.bfloat16)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                sensor_emb, vlm_emb, gate_logit = model(sensor_data, vision_features, text_features, is_embedded)
                
                clip_loss = siglip_loss(sensor_emb, vlm_emb, logit_scale.exp())
                
                # Use BCEWithLogitsLoss for numerical stability
                gate_loss = gate_criterion(gate_logit, contact_labels)

                total_loss = clip_loss + args.gate_loss_weight * gate_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1

            if is_main_process:
                current_lr = optimizer.param_groups[0]['lr']
                
                # Calculate gate accuracy for logging
                with torch.no_grad():
                    gate_prob = torch.sigmoid(gate_logit) # Convert logit to probability for accuracy calculation
                    gate_pred = (gate_prob > 0.5).float()
                    binary_labels = (contact_labels > 0.5).float()
                    gate_acc = (gate_pred == binary_labels).float().mean()

                pbar.set_postfix(loss=total_loss.item(), clip=clip_loss.item(), gate=gate_loss.item(), gate_acc=gate_acc.item(), lr=f"{current_lr:.2e}")

                wandb.log({
                    "train/loss": total_loss.item(),
                    "train/clip_loss": clip_loss.item(),
                    "train/gate_loss": gate_loss.item(),
                    "train/gate_acc": gate_acc.item(),
                    "train/lr": current_lr,
                    "global_step": global_step,
                    "epoch": epoch + 1
                })

        # Validation Loop
        model.eval()
        total_val_loss, total_val_clip_loss, total_val_gate_loss, total_val_gate_acc = 0, 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                sensor_data = batch["sensor_data"].to(device, dtype=torch.bfloat16)
                vision_features = batch["vision_features"].to(device, dtype=torch.bfloat16)
                text_features = batch["text_features"].to(device, dtype=torch.bfloat16)
                is_embedded = batch["is_embedded"]
                contact_labels = batch["contact_label"].to(device, dtype=torch.bfloat16)

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    sensor_emb, vlm_emb, gate_logit = model(sensor_data, vision_features, text_features, is_embedded)
                    clip_loss = siglip_loss(sensor_emb, vlm_emb, logit_scale.exp())
                    
                    # Use BCEWithLogitsLoss for numerical stability
                    gate_loss = gate_criterion(gate_logit, contact_labels)

                    total_loss = clip_loss + args.gate_loss_weight * gate_loss

                gate_prob = torch.sigmoid(gate_logit) # Convert logit to probability for accuracy calculation
                gate_pred = (gate_prob > 0.5).float()
                binary_labels = (contact_labels > 0.5).float()
                gate_acc = (gate_pred == binary_labels).float().mean()
                
                total_val_loss += total_loss.item()
                total_val_clip_loss += clip_loss.item()
                total_val_gate_loss += gate_loss.item()
                total_val_gate_acc += gate_acc.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_clip_loss = total_val_clip_loss / len(val_loader)
        avg_val_gate_loss = total_val_gate_loss / len(val_loader)
        avg_val_gate_acc = total_val_gate_acc / len(val_loader)

        if is_main_process:
            print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}, Val CLIP Loss: {avg_val_clip_loss:.4f}, Val Gate Loss: {avg_val_gate_loss:.4f}, Val Gate Acc: {avg_val_gate_acc:.4f}")

            wandb.log({
                "val/loss": avg_val_loss,
                "val/clip_loss": avg_val_clip_loss,
                "val/gate_loss": avg_val_gate_loss,
                "val/gate_acc": avg_val_gate_acc,
                "global_step": global_step,
                "epoch": epoch + 1
            })

            # Save checkpoint
            latest_checkpoint_path = Path(args.checkpoint_dir) / "sensor_clip_latest.pth"
            best_checkpoint_path = Path(args.checkpoint_dir) / "sensor_clip_best.pth"
            
            checkpoint_data = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.module.state_dict(),
                'encoder_state_dict': model.module.sensor_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'logit_scale': logit_scale.data,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss
            }

            torch.save(checkpoint_data, latest_checkpoint_path)
            print(f"💾 Latest model saved to {latest_checkpoint_path}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_data['best_val_loss'] = best_val_loss
                torch.save(checkpoint_data, best_checkpoint_path)
                print(f"✨ New best model saved to {best_checkpoint_path} with validation loss: {avg_val_loss:.4f}")


    if is_main_process:
        wandb.finish()

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sensor Encoder with CLIP and Auxiliary Gate Loss.")
    parser.add_argument('--new_dataset_paths', type=str, nargs='*', required=True)
    parser.add_argument('--cache_root', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--val_split', type=float, default=0.05)
    parser.add_argument('--gate_loss_weight', type=float, default=0.2, help='Weight for the auxiliary gate loss.')
    parser.add_argument('--contact_threshold', type=float, default=0.8, help='Temporal threshold to consider as contact.')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to a checkpoint to resume training from (e.g., "checkpoints/sensor_clip_latest.pth").')
    parser.add_argument('--sensor_noise_std', type=float, default=0.01, help='Standard deviation of Gaussian noise to add to sensor data for augmentation. Set to 0 to disable.')
    args = parser.parse_args()
    main(args)