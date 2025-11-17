"""
CLIP VLM Feature Cache Generation Script

This script generates a low-dimensional (512d) feature cache for CLIP-style
sensor encoder pre-training. It uses a 3B VLM (Qwen2.5-VL-3B-Instruct) and
an Attention Pooling mechanism to efficiently summarize vision information.

Methodology:
1.  Load the 3B VLM and a custom AttentionPooler module.
2.  Iterate through the specified datasets.
3.  For each sample (hand-eye camera view):
    a. Extract high-dimensional image tokens and text response from the VLM.
    b. Pass the image tokens through the AttentionPooler to get a 512d vision embedding.
    c. Pass the text response through a simple MLP to get a 512d text embedding.
4.  Save the (vision_embedding, text_embedding) tuple to a .pt cache file.

This pre-computation significantly speeds up the main CLIP training loop.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import hashlib
from pathlib import Path
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Add project root to import custom modules
import sys
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from vla_datasets.unified_dataset import create_unified_dataloader
from TRAIN_SensorImage_CLIP import (
    SensorImageCLIPDataset,
    get_formatted_clip_prompt,
    get_clip_prompt_hash,
    extract_task_name_from_episode_path,
    process_images_with_vlm,
    disable_generation_temperature
)
from vla_cache_manager import VLACacheManager
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# =====================================
# 1. Attention Pooling Module
# =====================================

class AttentionPooler(nn.Module):
    """
    Pools a sequence of image tokens into a fixed-size embedding using multi-head attention.
    A learnable query vector attends to the image tokens to summarize the information.
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
# 2. Main Caching Function
# =====================================

def main(args):
    # DDP Setup
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    is_main_process = local_rank == 0

    if is_main_process:
        print(f"Starting CLIP VLM cache generation with {dist.get_world_size()} GPUs.")
        print(f"Using VLM: {args.vlm_model}")
        print(f"Output embedding dimension: {args.embedding_dim}")

    # Load VLM
    vlm_processor = AutoProcessor.from_pretrained(args.vlm_model, trust_remote_code=True)
    vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.vlm_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2"
    )
    vlm_model.eval()
    disable_generation_temperature(vlm_model)
    for param in vlm_model.parameters():
        param.requires_grad = False

    # Attention Pooler and Text Projector
    vlm_hidden_size = vlm_model.config.text_config.hidden_size
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
    )
    
    clip_dataset = SensorImageCLIPDataset(
        unified_dataset,
        mode="cache_build",
    )

    sampler = DistributedSampler(clip_dataset, shuffle=False)
    dataloader = DataLoader(
        clip_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=lambda x: x # Keep as list of dicts
    )

    cache_manager = VLACacheManager(cache_dir=Path(args.cache_root) / "clip_vlm_features")

    pbar = tqdm(dataloader, desc="Generating Cache", disable=not is_main_process)
    for batch in pbar:
        for sample in batch:
            episode_id = sample["episode_id"]
            vlm_idx = sample["vlm_idx"]
            images = sample["hand_eye_image"] # List of PIL images

            task_name = extract_task_name_from_episode_path(Path(unified_dataset.datasets[0].data_dir.parent) / episode_id)
            prompt_hash = get_clip_prompt_hash(task_name)
            
            if cache_manager.cache_exists(episode_id, vlm_idx, prompt_hash):
                continue

            try:
                prompt = get_formatted_clip_prompt(task_name)
                with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    image_features, guidance_vector = process_images_with_vlm(
                        images, prompt, vlm_model, vlm_processor, device
                    )
                    
                    # Pool and project to low dimension
                    vision_embedding = vision_pooler(image_features.unsqueeze(0)).squeeze(0)
                    text_embedding = text_projector(guidance_vector.unsqueeze(0)).squeeze(0)

                features_to_cache = (
                    vision_embedding.detach().cpu().to(torch.float16),
                    text_embedding.detach().cpu().to(torch.float16)
                )
                cache_manager.save_cache_tuple(episode_id, vlm_idx, prompt_hash, features_to_cache)

            except Exception as e:
                if is_main_process:
                    print(f"Error processing {episode_id}_{vlm_idx}: {e}")

    if is_main_process:
        # Save the pooler and projector weights
        torch.save(vision_pooler.module.state_dict(), Path(args.checkpoint_dir) / "vision_pooler.pth")
        torch.save(text_projector.module.state_dict(), Path(args.checkpoint_dir) / "text_projector.pth")
        print("Cache generation complete. Pooler and projector weights saved.")

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate low-dimensional CLIP VLM feature cache.")
    parser.add_argument('--new_dataset_paths', type=str, nargs='*', required=True, help='Paths to the new format dataset directories.')
    parser.add_argument('--cache_root', type=str, required=True, help='Root directory for all caches.')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save pooler weights.')
    parser.add_argument('--vlm_model', type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help='VLM model name.')
    parser.add_argument('--embedding_dim', type=int, default=512, help='Output dimension for pooled features.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per GPU.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers.')
    args = parser.parse_args()
    main(args)