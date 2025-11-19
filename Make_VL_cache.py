import os
import sys
import math

from pathlib import Path
import hashlib, fcntl

from tqdm import tqdm

import torch
import torch.distributed as dist

from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader, DistributedSampler

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vla_datasets.unified_dataset import unified_collate_fn
from vla_cache_manager import get_cache_manager

# =====================================
# 1️⃣ Action Expert (Temporal Decoder)
# =====================================
def build_vl_cache_distributed_optimized(
    model,
    dataset,
    device="cuda",
    *,
    batch_size=16,          # DataLoader 배치 (VRAM 24GB면 높일 수 있음)
    num_workers=4,
    prefetch_factor=4,      # ✅ 4 → 8로 증가 (더 많이 미리 로드)
    micro_bs=1,            # 마이크로 배치 (OOM 시 자동 백오프)
    cache_dir_fallback="/home/najo/NAS/VLA/dataset/cache/qwen_vl_features",
):
    """
    완전 고정 캐싱 시스템 (VLACacheManager 사용):
      - 프롬프트 해시 기반 버전 관리
      - 마이크로배칭 + OOM 백오프
      - use_cache=False (KV cache 비활성화)
      - Atomic save + 캐시 용량 제한 자동 관리
      - tqdm 진행률, miss/skipped 통계 표시

    model 요구사항:
      - model.vl_model, model.processor 필요
      - (선택) model.cache_dir 있으면 사용, 없으면 cache_dir_fallback 사용
    """

    distributed = dist.is_available() and dist.is_initialized()
    if distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    # base cache dir
    base_cache_dir = getattr(model, "cache_dir", None)
    if base_cache_dir is None:
        base_cache_dir = Path(cache_dir_fallback)
    else:
        base_cache_dir = Path(base_cache_dir)

    # VLACacheManager 초기화
    cache_mgr = get_cache_manager(
        cache_dir=str(base_cache_dir),
        cache_limit_gb=50.0
    )

    # ---------------------------
    # DataLoader (샘플 분배 보장)
    # ---------------------------
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        shuffle=False if sampler else False,
        collate_fn=unified_collate_fn,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=True,              # ✅ GPU 전송 속도 향상
        persistent_workers=True,       # ✅ 워커 재사용으로 오버헤드 감소
    )

    total_local = math.ceil(len(dataset) / world_size)
    print(f"[Rank {rank}] Assigned ~{total_local} samples for caching.")
    current_device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    print(f"[Rank {rank}] CUDA ready: {torch.cuda.is_available()}, device={current_device}")

    # ---------------------------
    # 캐싱 루프
    # ---------------------------
    if hasattr(model, "eval"):
        model.eval()

    total_cached, total_skipped, total_processed = 0, 0, 0
    pending_keys = set()
    pbar = tqdm(
        total=total_local,
        desc=f"[Rank {rank}] Caching progress",
        dynamic_ncols=True,
        disable=(rank != 0)
    )

    with torch.inference_mode():
        for batch_idx, batch in enumerate(data_loader):
            texts = batch["instruction"]
            image_paths_list = batch["images"]
            cache_keys = batch["cache_keys"]
            vlm_indices = batch["vlm_indices"]
            prompt_hashes = batch["prompt_hash"]

            # --- 미스/스킵 분리 (VLACacheManager 사용) ---
            miss_items = []
            for i in range(len(cache_keys)):
                cache_key = cache_keys[i]
                vlm_idx = vlm_indices[i]
                prompt_hash = prompt_hashes[i]
                txt = texts[i]
                views = image_paths_list[i]

                dataset_name = cache_key.rsplit("_vlm", 1)[0]
                fingerprint = (dataset_name, int(vlm_idx), prompt_hash)
                if fingerprint in pending_keys:
                    total_skipped += 1
                    continue
                if not cache_mgr.cache_exists(dataset_name, vlm_idx, prompt_hash):
                    if views:
                        miss_items.append({
                            "text": txt,
                            "views": views,
                            "dataset_name": dataset_name,
                            "vlm_idx": vlm_idx,
                            "prompt_hash": prompt_hash,
                        })
                        pending_keys.add(fingerprint)
                    else:
                        total_skipped += 1
                else:
                    total_skipped += 1

            total_processed += len(cache_keys)
            if not miss_items:
                pbar.update(len(cache_keys))
                if rank == 0:
                    cached_ratio = (total_cached / max(1, total_processed)) * 100 if total_processed > 0 else 0
                    pbar.set_postfix({
                        "cached": total_cached,
                        "skipped": total_skipped,
                        "miss%": f"{100 - cached_ratio:.1f}%",
                        "GPU": f"{torch.cuda.memory_allocated(device)/1e9:.1f}GB"
                    })
                continue

            # --- 메시지 전처리 (CPU) ---
            messages_list = []
            for item in miss_items:
                txt, views = item["text"], item["views"]
                msg_content = [{"type": "image", "image": v} for v in views if v is not None]
                msg_content.append({"type": "text", "text": txt})
                messages_list.append([{"role": "user", "content": msg_content}])

            processed_texts, vision_inputs_list = [], []
            for messages in messages_list:
                text = model.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                vision_inputs, _ = process_vision_info(messages)
                
                if vision_inputs is None:
                    vision_inputs = []
                    
                processed_texts.append(text)
                vision_inputs_list.append(vision_inputs)

            # --- 마이크로배칭 + OOM 백오프 ---
            start = 0
            _micro_bs = max(1, micro_bs)
            while start < len(miss_items):
                end = min(start + _micro_bs, len(miss_items))
                sub_items = miss_items[start:end]

                try:
                    # 각 아이템에 대해 이미지/텍스트 분리 추론 수행
                    for item in sub_items:
                        txt, views = item["text"], item["views"]

                        # 1. 이미지 전용 추론 (순수 이미지 특징 추출)
                        if views:
                            msg_content = [{"type": "image", "image": v} for v in views]
                            msg_content.append({"type": "text", "text": ""}) # 빈 텍스트
                            messages = [{"role": "user", "content": msg_content}]
                            text_with_placeholders = model.processor.apply_chat_template(
                                messages, tokenize=False, add_generation_prompt=False
                            )
                            image_only_inputs = model.processor(
                                text=[text_with_placeholders], images=views, padding=True, return_tensors="pt"
                            ).to(device=device, dtype=torch.bfloat16, non_blocking=True)
                            
                            image_outputs = model.vl_model(**image_only_inputs, output_hidden_states=True, return_dict=True)
                            image_hidden_state = image_outputs.hidden_states[-1]

                            # ✅ FIX: Extract image patches between <|vision_start|> and <|vision_end|> markers
                            # Qwen2.5-VL special tokens:
                            # <|vision_start|> = 151652
                            # <|vision_end|> = 151653
                            # <|image_pad|> (151655) is expanded into actual image patch embeddings by VLM
                            input_ids_flat = image_only_inputs['input_ids'].squeeze(0)
                            vision_start_positions = torch.where(input_ids_flat == 151652)[0]
                            vision_end_positions = torch.where(input_ids_flat == 151653)[0]

                            # Collect all image patch tokens between vision markers
                            image_patch_indices = []
                            for start_pos, end_pos in zip(vision_start_positions, vision_end_positions):
                                patch_indices = torch.arange(start_pos + 1, end_pos, device=input_ids_flat.device)
                                image_patch_indices.append(patch_indices)

                            if image_patch_indices:
                                image_patch_indices = torch.cat(image_patch_indices)
                                image_features = image_hidden_state[:, image_patch_indices, :]
                            else:
                                image_features = torch.empty(1, 0, model.vl_model.config.hidden_size, device=device, dtype=torch.bfloat16)
                        else:
                            image_features = torch.empty(1, 0, model.vl_model.config.hidden_size, device=device, dtype=torch.bfloat16)

                        # 2. 텍스트 전용 추론 (가이던스 벡터 추출)
                        if txt:
                            text_only_inputs = model.processor(
                                text=[txt], images=None, padding=True, return_tensors="pt"
                            ).to(device=device, dtype=torch.bfloat16, non_blocking=True)
                            
                            text_outputs = model.vl_model(**text_only_inputs, output_hidden_states=True, return_dict=True)
                            text_hidden_state = text_outputs.hidden_states[-1]
                            guidance_vector = text_hidden_state.mean(dim=1)
                        else:
                            guidance_vector = torch.zeros(1, model.vl_model.config.hidden_size, device=device, dtype=torch.bfloat16)

                        # 3. 캐시 저장 (튜플 형식으로)
                        features_to_cache = (
                            image_features.detach().to("cpu", dtype=torch.float16),
                            guidance_vector.detach().to("cpu", dtype=torch.float16)
                        )
                        cache_mgr.save_cache_tuple(
                            dataset_name=item["dataset_name"],
                            vlm_idx=item["vlm_idx"],
                            prompt_hash=item["prompt_hash"],
                            features_tuple=features_to_cache
                        )
                        total_cached += 1

                    start = end

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        torch.cuda.empty_cache()
                        if _micro_bs == 1:
                            print(f"[Rank {rank}] ❌ OOM on micro_bs=1 for item: {sub_items[0]['dataset_name']}_{sub_items[0]['vlm_idx']}. Skipping.")
                            start += 1 # 현재 아이템 건너뛰기
                            continue
                        _micro_bs = max(1, _micro_bs // 2)
                        if rank == 0:
                            print(f"⚠️ [OOM] Lowering micro_bs to #{_micro_bs} and retrying...")
                        continue
                    else:
                        raise

            # --- 진행률 업데이트 ---
            pbar.update(len(cache_keys))
            if rank == 0:
                cached_ratio = (total_cached / max(1, total_processed)) * 100
                pbar.set_postfix({
                    "cached": total_cached,
                    "skipped": total_skipped,
                    "miss%": f"{100 - cached_ratio:.1f}%",
                    "GPU": f"{torch.cuda.memory_allocated(device)/1e9:.1f}GB"
                })

    pbar.close()
    print(f"[Rank {rank}] ✅ Finished. Cached {total_cached} / Skipped {total_skipped}")
    dist.barrier()
    if rank == 0:
        print("🚀 All ranks finished caching. Cache is ready for training.")
