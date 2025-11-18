"""
CLIP VLM Feature Cache Generation Script (Using QwenVLAUnified)

This script generates low-dimensional (512d) feature cache for CLIP-style
sensor encoder pre-training. It uses the QwenVLAUnified model's VL encoder
to extract vision and text features, then compresses them using attention pooling.

Based on Make_VL_cache.py methodology but with attention pooling compression.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import sys

# Add project root to import custom modules
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.unified_model import QwenVLAUnified
from vla_datasets.unified_dataset import create_unified_dataloader, unified_collate_fn
from vla_cache_manager import VLACacheManager
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader, DistributedSampler

# =====================================
# Attention Pooling Module
# =====================================

class AttentionPooler(nn.Module):
    """
    Pools a sequence of image tokens into a fixed-size embedding using multi-head attention.
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
# Main Caching Function
# =====================================

def main(args):
    # DDP Setup
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_main_process = local_rank == 0

    if is_main_process:
        print(f"Starting CLIP VLM cache generation with {world_size} GPUs.")
        print(f"Using VLM model with QwenVLAUnified")
        print(f"Output embedding dimension: {args.embedding_dim}")

    # Load VLM using QwenVLAUnified (this loads the model correctly)
    model = QwenVLAUnified(
        model_type='flow_matching',
        vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        action_dim=7,
        horizon=8,
        hidden_dim=1024,
        sensor_enabled=False,  # Don't need sensor for cache building
        finetune_vl='none',
        image_resize_height=args.image_resize_height,
        image_resize_width=args.image_resize_width,
        device_map=None,
        external_cache_root=args.cache_root,
    ).to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    # Attention Pooler and Text Projector
    vlm_hidden_size = model.vl_model.config.hidden_size
    vision_pooler = AttentionPooler(vlm_hidden_size, args.embedding_dim).to(device, dtype=torch.bfloat16)
    text_projector = nn.Linear(vlm_hidden_size, args.embedding_dim).to(device, dtype=torch.bfloat16)

    # DDP Wrapping
    vision_pooler = DDP(vision_pooler, device_ids=[local_rank])
    text_projector = DDP(text_projector, device_ids=[local_rank])

    # Dataset
    unified_dataset = create_unified_dataloader(
        new_dataset_paths=args.new_dataset_paths,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        return_dataset=True,
        use_cache=False,
        skip_dataset_stats=True,
        cache_build_only=True,
    )

    # DataLoader with DistributedSampler
    sampler = DistributedSampler(unified_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(
        unified_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=unified_collate_fn,
        pin_memory=True,
    )

    cache_manager = VLACacheManager(cache_dir=Path(args.cache_root) / "clip_vlm_features")

    total_cached = 0
    total_skipped = 0

    pbar = tqdm(dataloader, desc="Generating Cache", disable=not is_main_process)

    with torch.inference_mode():
        for batch in pbar:
            texts = batch["instruction"]
            image_paths_list = batch["images"]
            cache_keys = batch["cache_keys"]
            vlm_indices = batch["vlm_indices"]
            prompt_hashes = batch["prompt_hash"]

            # Process each sample in the batch
            for i in range(len(cache_keys)):
                cache_key = cache_keys[i]
                vlm_idx = vlm_indices[i]
                prompt_hash = prompt_hashes[i]
                txt = texts[i]
                views = image_paths_list[i]

                # Extract dataset_name from cache_key (format: "dataset_name_vlmX")
                dataset_name = cache_key.rsplit("_vlm", 1)[0]

                # Check if cache already exists
                if cache_manager.cache_exists(dataset_name, vlm_idx, prompt_hash):
                    total_skipped += 1
                    continue

                # Skip if no images
                if not views or len(views) == 0:
                    total_skipped += 1
                    if is_main_process:
                        print(f"Skipping {dataset_name}_{vlm_idx}: No images available")
                    continue

                try:
                    # 1. Image-only inference (extract pure image features)
                    msg_content = [{"type": "image", "image": v} for v in views]
                    msg_content.append({"type": "text", "text": ""})  # Empty text
                    messages = [{"role": "user", "content": msg_content}]

                    text_with_placeholders = model.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                    image_only_inputs = model.processor(
                        text=[text_with_placeholders], images=views, padding=True, return_tensors="pt"
                    ).to(device=device, dtype=torch.bfloat16, non_blocking=True)

                    image_outputs = model.vl_model(**image_only_inputs, output_hidden_states=True, return_dict=True)
                    image_hidden_state = image_outputs.hidden_states[-1]

                    # Extract image tokens (token ID 151857 is <|image_pad|>)
                    image_token_mask = (image_only_inputs['input_ids'] == 151857)
                    image_indices = torch.where(image_token_mask.any(dim=0))[0]
                    if len(image_indices) > 0:
                        image_features = image_hidden_state[:, image_indices, :]  # (1, N_tokens, D)
                    else:
                        # Fallback: use all tokens
                        image_features = image_hidden_state

                    # 2. Text-only inference (extract guidance vector)
                    if txt:
                        text_only_inputs = model.processor(
                            text=[txt], images=None, padding=True, return_tensors="pt"
                        ).to(device=device, dtype=torch.bfloat16, non_blocking=True)

                        text_outputs = model.vl_model(**text_only_inputs, output_hidden_states=True, return_dict=True)
                        text_hidden_state = text_outputs.hidden_states[-1]
                        guidance_vector = text_hidden_state.mean(dim=1)  # (1, D)
                    else:
                        guidance_vector = torch.zeros(1, vlm_hidden_size, device=device, dtype=torch.bfloat16)

                    # 3. Pool and project to low dimension
                    vision_embedding = vision_pooler(image_features).squeeze(0)  # (D_out,)
                    text_embedding = text_projector(guidance_vector).squeeze(0)  # (D_out,)

                    # 4. Save cache
                    features_to_cache = (
                        vision_embedding.detach().cpu().to(torch.float16),
                        text_embedding.detach().cpu().to(torch.float16)
                    )
                    cache_manager.save_cache_tuple(dataset_name, vlm_idx, prompt_hash, features_to_cache)
                    total_cached += 1

                except Exception as e:
                    if is_main_process:
                        print(f"Error processing {dataset_name}_{vlm_idx}: {e}")
                    continue

            if is_main_process:
                pbar.set_postfix({
                    "cached": total_cached,
                    "skipped": total_skipped,
                })

    if is_main_process:
        # Save the pooler and projector weights
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(vision_pooler.module.state_dict(), checkpoint_dir / "vision_pooler.pth")
        torch.save(text_projector.module.state_dict(), checkpoint_dir / "text_projector.pth")
        print(f"Cache generation complete. Cached: {total_cached}, Skipped: {total_skipped}")
        print("Pooler and projector weights saved.")

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate low-dimensional CLIP VLM feature cache.")
    parser.add_argument('--new_dataset_paths', type=str, nargs='*', required=True, help='Paths to the new format dataset directories.')
    parser.add_argument('--cache_root', type=str, required=True, help='Root directory for all caches.')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save pooler weights.')
    parser.add_argument('--embedding_dim', type=int, default=512, help='Output dimension for pooled features.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per GPU.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers.')
    parser.add_argument('--image_resize_height', type=int, default=360, help='Image resize height.')
    parser.add_argument('--image_resize_width', type=int, default=640, help='Image resize width.')
    args = parser.parse_args()
    main(args)
