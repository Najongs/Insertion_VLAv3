"""
Cache VLM Features for CLIP Pre-training - DDP Version

This script pre-computes and caches features from the VLM for the samples
that will be used in `TRAIN_SensorImage_CLIP.py`.

This version uses torch.distributed (DDP) to parallelize the workload across
multiple GPUs, loading one VLM per GPU to avoid memory issues.

Usage:
    torchrun --nproc_per_node=4 cache_clip_vlm_features.py \
        --new_dataset_paths "/path/to/dataset1" "/path/to/dataset2" \
        --cache_root "/path/to/cache" \
        --vlm_model "Qwen/Qwen2.5-VL-7B-Instruct"
"""

# =============================================================================
# ⚠️ CRITICAL: CLIP VLM 캐시 구조 및 prompt_hash 매칭 (Updated 2025-01-13)
# =============================================================================
# 이 스크립트는 CLIP 학습용 VLM 캐시를 생성합니다.
# 캐시 구조를 올바르게 이해하는 것이 매우 중요합니다.
#
# 1. CLIP VLM 캐시 경로 구조 (2025-01-13 업데이트):
#    {cache_root}/clip_vlm_features/{prompt_hash}/{episode_name}_vlm{idx}.pt
#
#    - prompt_hash: CLIP_PROMPT_TEXT를 task_name으로 포맷한 후 MD5 해시화한 값 (첫 8자)
#    - CLIP_PROMPT_TEXT는 템플릿 문자열이며 {task_name} 플레이스홀더를 포함
#    - 각 태스크(Red_point, White_point, etc.)가 고유한 prompt_hash를 가짐
#    - 각 태스크의 캐시가 별도의 prompt_hash 디렉토리에 저장됨
#
# 2. 2025-01-13 주요 변경사항:
#    ✨ CLIP도 이제 task-specific prompts 사용:
#       - CLIP_PROMPT_TEXT 템플릿: "...target point is {task_name}..."
#       - Red_point  → "...target point is Red_point..."  → hash_red
#       - White_point → "...target point is White_point..." → hash_white
#       - 캐시 구조: /cache/clip_vlm_features/hash_red/, /cache/clip_vlm_features/hash_white/
#
#    🖼️ Single-view support:
#       - VLM에 View5 카메라 뷰의 이미지만 전달
#       - Content structure: [{"type": "image", "image": img}, {"type": "text", "text": prompt}]
#
# 3. Flow Matching VL 캐시와의 차이점:
#
#    ┌─────────────────┬──────────────────────────┬─────────────────────────────┐
#    │                 │  CLIP VLM 캐시           │  Flow Matching VL 캐시      │
#    ├─────────────────┼──────────────────────────┼─────────────────────────────┤
#    │ Prompt 소스     │ CLIP_PROMPT_TEXT template│ 태스크별 instruction        │
#    │ 해시 알고리즘   │ MD5                      │ SHA256                      │
#    │ task_name 포함  │ O (태스크별로 다름)      │ O (태스크별로 다름)         │
#    │ 캐시 디렉토리   │ clip_vlm_features/       │ qwen_vl_features/           │
#    │ prompt_hash 수  │ N개 (태스크당 1개)       │ N개 (태스크당 1개)          │
#    │ 이미지 입력     │ Single view (View5)      │ Single view                 │
#    └─────────────────┴──────────────────────────┴─────────────────────────────┘
#
# 4. ⚠️ 캐시 무효화 주의사항:
#    - CLIP_PROMPT_TEXT 템플릿을 변경하면 모든 prompt_hash가 바뀌어 기존 캐시를 찾을 수 없음
#    - CLIP_PROMPT_TEXT 변경 시 반드시 모든 태스크의 캐시를 재생성해야 함
#    - 학습 시 TRAIN_SensorImage_CLIP.py의 CLIP_PROMPT_TEXT와 완전히 일치해야 함
#
# 5. 캐시 생성과 학습 간 일관성:
#    ✅ 반드시 확인해야 할 사항:
#       - 동일한 CLIP_PROMPT_TEXT 템플릿 사용 (TRAIN_SensorImage_CLIP.py에서 import)
#       - 동일한 cache_root 경로
#       - 동일한 VLM 모델 (Qwen2.5-VL-3B-Instruct 등)
#       - 동일한 hand-eye view 설정 (View5 only)
# =============================================================================

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Subset, DistributedSampler
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import time

# Add project root to import custom modules
import sys
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from vla_datasets.unified_dataset import create_unified_dataloader
from TRAIN_SensorImage_CLIP import (
    SensorImageCLIPDataset,
    CLIP_PROMPT_TEXT,
    get_clip_prompt_hash,
    get_formatted_clip_prompt,
    extract_task_name_from_episode_path,
)
from qwen_vl_utils import process_vision_info
from vla_cache_manager import VLACacheManager


def disable_generation_temperature(vlm_model):
    """
    Keep the temperature attribute but neutralize it so the model
    does not try to use it (while avoiding HF warnings).
    """
    gen_cfg = getattr(vlm_model, "generation_config", None)
    if gen_cfg is None:
        return

    extra_params = getattr(gen_cfg, "_extra_generation_params", None)
    if isinstance(extra_params, dict):
        extra_params.pop("temperature", None)

    try:
        setattr(gen_cfg, "temperature", None)
    except AttributeError:
        gen_cfg.__dict__["temperature"] = None


def setup_distributed():
    """Initialize distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return 0, 1, 0, False

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    return rank, world_size, local_rank, True


def cleanup_distributed():
    """Cleanup distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def generate_text_response(vlm_model, vlm_processor, messages, max_new_tokens):
    # messages: [{"role":"user","content":[{"type":"image","image": raw_image}, {"type":"text","text": prompt}]}]
    text_input = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    model_inputs = vlm_processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device=vlm_model.device, dtype=vlm_model.dtype)

    input_lens = [len(ids) for ids in model_inputs.input_ids]
    with torch.no_grad():
        gen_ids = vlm_model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )

    trimmed = [ids[il:] for ids, il in zip(gen_ids, input_lens)]
    resp = vlm_processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    # Qwen 템플릿 구간 제거
    if "<|im_start|>assistant" in resp:
        resp = resp.split("<|im_start|>assistant", 1)[-1]
    if "<|im_end|>" in resp:
        resp = resp.split("<|im_end|>", 1)[0]
    return resp.strip()



def cache_worker(rank, world_size, local_rank, args, clip_dataset):
    """
    The worker function for each DDP process.
    Loads a model onto its assigned GPU and processes a subset of the data.
    """
    device = torch.device(f"cuda:{local_rank}")
    is_main_process = rank == 0

    # Each worker gets its own cache manager instance
    cache_manager = VLACacheManager(cache_dir=str(Path(args.cache_root) / "clip_vlm_features"))

    # Build episode_id -> task_name mapping from unified_dataset
    episode_to_task = {}
    for sub_dataset in clip_dataset.unified_dataset.datasets:
        episode_id = sub_dataset.data_dir.name
        task_name = extract_task_name_from_episode_path(sub_dataset.data_dir)
        episode_to_task[episode_id] = task_name

    if is_main_process:
        print(f"[Rank {rank}] Built task name mapping for {len(episode_to_task)} episodes")
        unique_tasks = set(episode_to_task.values())
        print(f"[Rank {rank}] Found {len(unique_tasks)} unique tasks: {sorted(unique_tasks)}")

    # 1. Load VLM (one per process)
    if is_main_process:
        print(f"[Rank {rank}] Loading VLM on GPU {local_rank}...")

    vlm_processor = AutoProcessor.from_pretrained(
        args.vlm_model,
        trust_remote_code=True,
        local_files_only=True  # Use local cache only, avoid network verification
    )
    vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.vlm_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": device},  # Load entire model on this specific GPU
        attn_implementation="flash_attention_2",
        local_files_only=True  # Use local cache only, avoid network verification
    )
    vlm_model.eval()
    disable_generation_temperature(vlm_model)

    if is_main_process:
        print(f"[Rank {rank}] VLM loaded on {device}.")

    # 2. Create DistributedSampler for this worker
    sampler = DistributedSampler(
        clip_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    )

    def collate_fn_cache(batch):
        """Collate function that keeps samples as list for batch processing"""
        return batch

    dataloader = DataLoader(
        clip_dataset,
        batch_size=args.batch_size,  # Process in batches
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn_cache
    )

    # 3. Iterate and cache features
    if is_main_process:
        pbar = tqdm(total=len(dataloader), desc=f"Rank {rank} (GPU {local_rank})")

    for batch in dataloader:
        # Process each sample in the batch
        for sample in batch:
            images = sample["hand_eye_image"]  # Now a list of images (View5, View4, etc.)
            episode_id = sample["episode_id"]
            vlm_idx = sample["vlm_idx"]

            if vlm_idx is None:
                continue

            # Get task-specific prompt and hash
            task_name = episode_to_task.get(episode_id, "Unknown")
            formatted_prompt = get_formatted_clip_prompt(task_name)
            task_prompt_hash = get_clip_prompt_hash(task_name)

            if cache_manager.cache_exists(dataset_name=episode_id, vlm_idx=vlm_idx, prompt_hash=task_prompt_hash):
                continue

            # Generate text, image embeds, and text embeds
            try:
                # Use only the first image (View5)
                image = images[0] if isinstance(images, list) and len(images) > 0 else images
                messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": formatted_prompt}]}]
                text_response = generate_text_response(
                    vlm_model, vlm_processor, messages, args.max_new_tokens
                )

                with torch.no_grad():
                    # 1. 이미지 전용 추론 (순수 이미지 특징 추출 - 모든 이미지 토큰)
                    image_only_messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": ""}]}]
                    image_text_with_placeholders = vlm_processor.apply_chat_template(
                        image_only_messages, tokenize=False, add_generation_prompt=False
                    )
                    image_only_vision_input, _ = process_vision_info(image_only_messages)
                    image_inputs = vlm_processor(
                        text=[image_text_with_placeholders],
                        images=image_only_vision_input, padding=True, return_tensors="pt"
                    ).to(device=vlm_model.device, dtype=vlm_model.dtype)

                    image_outputs = vlm_model(**image_inputs, output_hidden_states=True, return_dict=True)
                    image_hidden_state = image_outputs.hidden_states[-1]

                    # 이미지 토큰만 추출 (토큰 ID 151655 for Qwen2.5-VL image_pad)
                    image_token_mask = (image_inputs['input_ids'] == 151655)
                    image_indices = torch.where(image_token_mask.squeeze(0))[0]
                    image_features = image_hidden_state[:, image_indices, :]

                    # 2. 텍스트 전용 추론 (가이던스 벡터 추출 - 평균 풀링)
                    text_inputs = vlm_processor(
                        text=[text_response], images=None, padding=True, return_tensors="pt"
                    ).to(device=vlm_model.device, dtype=vlm_model.dtype)

                    text_outputs = vlm_model(**text_inputs, output_hidden_states=True, return_dict=True)
                    text_hidden_state = text_outputs.hidden_states[-1]
                    guidance_vector = text_hidden_state.mean(dim=1)

                # 3. 캐시 저장 (튜플 형식으로, task-specific prompt hash 사용)
                features_to_cache = (
                    image_features.detach().to("cpu", dtype=torch.float16),
                    guidance_vector.detach().to("cpu", dtype=torch.float16)
                )

                cache_manager.save_cache_tuple(
                    dataset_name=episode_id, vlm_idx=vlm_idx, prompt_hash=task_prompt_hash, features_tuple=features_to_cache
                )

            except Exception as e:
                if is_main_process:
                    print(f"[Rank {rank}] Error processing {episode_id}_vlm{vlm_idx}: {e}")

        if is_main_process:
            pbar.update(1)

    if is_main_process:
        pbar.close()


def main():
    parser = argparse.ArgumentParser(description="Cache VLM features for CLIP pre-training (DDP Version).")
    parser.add_argument('--new_dataset_paths', type=str, nargs='*',
                       default=["/home/najo/NAS/VLA/dataset/New_dataset", "/home/najo/NAS/VLA/dataset/New_dataset2"])
    parser.add_argument('--old_dataset_patterns', type=str, nargs='*', default=[])
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for dataloader (higher = faster but more memory).')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of dataloader workers per GPU process.')
    parser.add_argument('--vlm_model', type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                       help='VLM model for encoding.')
    parser.add_argument('--cache_root', type=str, default="/home/najo/NAS/VLA/dataset/cache",
                       help='Root directory for all caches.')
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--skip_dataset_stats', action='store_true',
                       help='Skip dataset statistics collection for faster startup')

    args = parser.parse_args()

    # Setup distributed environment
    rank, world_size, local_rank, is_distributed = setup_distributed()

    if rank == 0:
        print(f"🚀 Starting CLIP VLM feature caching with {world_size} GPUs")
        print(f"📂 Dataset paths: {args.new_dataset_paths}")
        print(f"💾 Cache root: {args.cache_root}")
        print(f"🤖 VLM model: {args.vlm_model}")
        print(f"📦 Batch size: {args.batch_size} (per GPU)")
        print(f"👷 Workers: {args.num_workers} (per GPU)")
        print()

    # Create dataset (only on rank 0, then broadcast)
    if rank == 0:
        print("📊 Creating dataset to identify all valid samples for caching...")
        unified_dataset = create_unified_dataloader(
            new_dataset_paths=args.new_dataset_paths,
            old_dataset_patterns=args.old_dataset_patterns,
            return_dataset=True,
            use_cache=False,
            skip_dataset_stats=args.skip_dataset_stats,
        )

        # This dataset filters for the last 20% of samples etc.
        clip_dataset = SensorImageCLIPDataset(
            unified_dataset,
            vlm_annotations={},
            use_augmentation=False,
            mode="cache_build"
        )
        print(f"✅ Found {len(clip_dataset)} total valid samples to process.")
        print()
    else:
        # Other ranks: wait for rank 0 to create dataset
        unified_dataset = create_unified_dataloader(
            new_dataset_paths=args.new_dataset_paths,
            old_dataset_patterns=args.old_dataset_patterns,
            return_dataset=True,
            use_cache=False,
            skip_dataset_stats=args.skip_dataset_stats,
        )
        clip_dataset = SensorImageCLIPDataset(
            unified_dataset,
            vlm_annotations={},
            use_augmentation=False,
            mode="cache_build"
        )

    # Synchronize all processes
    if is_distributed:
        dist.barrier()

    # Run caching worker
    cache_worker(rank, world_size, local_rank, args, clip_dataset)

    # Synchronize before final stats
    if is_distributed:
        dist.barrier()

    # Print final stats (only rank 0)
    if rank == 0:
        print("\n✅ VLM feature caching complete.")
        cache_manager = VLACacheManager(cache_dir=str(Path(args.cache_root) / "clip_vlm_features"))
        stats = cache_manager.get_cache_stats()
        print("📊 Cache statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")

    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
