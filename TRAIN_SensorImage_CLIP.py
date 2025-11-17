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

# Add project root to import custom modules
import sys
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.Encoder_model import UnifiedGatedSensorEncoder
from vla_datasets.unified_dataset import create_unified_dataloader
from vla_cache_manager import VLACacheManager

# =====================================
# 1. Dataset with Time-based Pseudo-Labeling
# =====================================

class SensorCLIPDataset(Dataset):
    def __init__(self, unified_dataset, clip_cache_root: str, contact_threshold: float = 0.8):
        self.unified_dataset = unified_dataset
        self.clip_cache_manager = VLACacheManager(cache_dir=clip_cache_root)
        self.contact_threshold = contact_threshold

        self.samples = []
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
                
                self.samples.append({
                    "global_idx": global_idx,
                    "relative_pos": relative_pos,
                    "contact_label": 1.0 if relative_pos >= self.contact_threshold else 0.0
                })
            global_idx_offset += episode_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        try:
            data = self.unified_dataset[sample_info["global_idx"]]
            
            # Load from low-dimensional cache
            episode_id = data["episode_id"]
            vlm_idx = data["vlm_idx"]
            # NOTE: Assuming a single prompt hash for simplicity now.
            # This can be extended to be task-specific if needed.
            prompt_hash = "low_dim_cache" 
            
            vision_embedding, text_embedding = self.clip_cache_manager.load_cache(
                episode_id, vlm_idx, prompt_hash
            )

            return {
                "sensor_data": data["sensor_data"],
                "vision_embedding": vision_embedding,
                "text_embedding": text_embedding,
                "contact_label": sample_info["contact_label"]
            }
        except Exception as e:
            # On error, return the first valid sample
            # print(f"Warning: Error loading sample {idx}. Returning first sample. Error: {e}")
            return self.__getitem__(0)

# =====================================
# 2. CLIP Model Definition
# =====================================

class CLIPModel(nn.Module):
    def __init__(self, sensor_encoder: UnifiedGatedSensorEncoder, embedding_dim: int = 512, nhead: int = 8):
        super().__init__()
        self.sensor_encoder = sensor_encoder
        
        # VLM fusion module
        self.vlm_fusion = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=nhead,
            batch_first=True
        )
        self.vlm_norm = nn.LayerNorm(embedding_dim)

    def forward(self, sensor_data, vision_embedding, text_embedding):
        # Sensor encoding
        sensor_embedding, gate_logit = self.sensor_encoder(sensor_data)
        
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
    model = CLIPModel(sensor_encoder, embedding_dim=args.embedding_dim).to(device)
    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
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

    # Split and create DataLoaders
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)

    # Scheduler and Loss
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * args.epochs, eta_min=1e-6)
    gate_criterion = nn.BCEWithLogitsLoss()
    logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07))).to(device)

    # Training Loop
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not is_main_process)
        
        for batch in pbar:
            sensor_data = batch["sensor_data"].to(device, dtype=torch.bfloat16)
            vision_emb = batch["vision_embedding"].to(device, dtype=torch.bfloat16)
            text_emb = batch["text_embedding"].to(device, dtype=torch.bfloat16)
            contact_labels = batch["contact_label"].to(device, dtype=torch.bfloat16)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                sensor_emb, vlm_emb, gate_logit = model(sensor_data, vision_emb, text_emb)
                
                # CLIP Loss
                clip_loss = siglip_loss(sensor_emb, vlm_emb, logit_scale.exp())
                
                # Gate Loss
                gate_loss = gate_criterion(gate_logit, contact_labels)
                
                total_loss = clip_loss + args.gate_loss_weight * gate_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            if is_main_process:
                pbar.set_postfix(loss=total_loss.item(), clip=clip_loss.item(), gate=gate_loss.item())

        # Validation Loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                sensor_data = batch["sensor_data"].to(device, dtype=torch.bfloat16)
                vision_emb = batch["vision_embedding"].to(device, dtype=torch.bfloat16)
                text_emb = batch["text_embedding"].to(device, dtype=torch.bfloat16)
                contact_labels = batch["contact_label"].to(device, dtype=torch.bfloat16)

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    sensor_emb, vlm_emb, gate_logit = model(sensor_data, vision_emb, text_emb)
                    clip_loss = siglip_loss(sensor_emb, vlm_emb, logit_scale.exp())
                    gate_loss = gate_criterion(gate_logit, contact_labels)
                    total_loss = clip_loss + args.gate_loss_weight * gate_loss
                
                total_val_loss += total_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        if is_main_process:
            print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")
            torch.save(model.module.sensor_encoder.state_dict(), Path(args.checkpoint_dir) / "sensor_clip_latest.pth")

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
    args = parser.parse_args()
    main(args)