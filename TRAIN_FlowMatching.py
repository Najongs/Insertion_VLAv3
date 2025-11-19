"""
Flow Matching VLA Training Script with Sensor Integration

**최근 변경 사항 및 주요 개념 (2025-11-11):**

1.  **이중 캐싱 전략 (Dual Caching Strategy):**
    -   **CLIP 센서 사전학습 캐시:** `cache_clip_vlm_features.py`를 통해 생성됩니다. 이 캐시는 의도적으로 각 에피소드 데이터의 마지막 20%만 사용합니다. 이는 센서의 접촉 이벤트가 주로 에피소드 후반부에 발생하기 때문입니다.
    -   **Action Decoder 학습 캐시:** 이 스크립트(`TRAIN_FlowMatching.py --mode cache`)를 통해 생성됩니다. 메인 VLA 모델의 효과적인 학습을 위해, 이 캐시는 데이터셋의 100% 전체를 포함하는 것이 매우 중요합니다.

2.  **100% 캐싱 문제 해결:**
    -   **문제:** Action Decoder 캐싱 시 전체 데이터셋이 아닌 일부만 처리되는 문제가 있었습니다.
    -   **원인:** `vlm_reuse_count` 파라미터가 원인이었습니다. 이 값은 여러 프레임에 걸쳐 VLM 피처를 얼마나 재사용할지 결정하며, 주로 실시간 추론 최적화를 위해 사용됩니다. 기본값 3으로 인해 전체 프레임의 약 1/3만 캐시 대상으로 고려되었습니다.
    -   **해결:** `--vlm_reuse_count` 커맨드라인 인자를 추가했습니다. 캐시 생성 시에는 `TOTAL_TRAIN.sh`에서 이 값을 `1`로 명시적으로 설정하여, 모든 프레임이 고유한 캐시를 갖도록 보장하고 100% 캐싱을 수행하도록 수정했습니다.

3.  **`fusion_strategy` 인자 관련:**
    -   `QwenVLAUnified` 모델 초기화 시 `fusion_strategy` 관련 `TypeError`가 발생했던 문제를 해결했습니다. 이 인자는 향후 다양한 센서 융합 전략을 구현하기 위해 예약된 것으로, 현재 모델에서는 받지 않으므로 일시적으로 제거되었습니다.

4.  **⚠️ VL 캐시 경로 구조와 prompt_hash의 중요성 (2025-01-12):**
    -   **캐시 경로 구조:** VL 캐시는 `{cache_root}/{prompt_hash}/{episode_name}_vlm{idx}.pt` 형태로 저장됩니다.
    -   **prompt_hash 생성:** instruction 텍스트를 SHA256 해시화한 값의 첫 8자입니다.
    -   **핵심 문제:**
        * new_format_dataset.py에서 task_name (예: "Red point", "Blue point")이 instruction에 포함됨
        * 태스크마다 다른 instruction → 다른 prompt_hash → 별도의 캐시 디렉토리
        * 예: Red_point → prompt_hash a1b2c3d4, Blue_point → prompt_hash e5f6g7h8
    -   **발생했던 문제:**
        * 기존 캐시는 Red_point만 생성되어 있었음 (92943a2d 디렉토리)
        * 다른 태스크(Blue/Green/White/Yellow)는 캐시 없음
        * cache_only_mode에서 캐시 없는 샘플 → VLM 실시간 생성 시도 → VLM 없음 → 에러!
    -   **해결 방법:**
        * 모든 태스크에 대해 캐시 생성 (TOTAL_TRAIN.sh의 캐시 생성 단계)
        * vlm_reuse_count=3으로 통일 (캐시 생성 & 학습)
        * prompt_hash_override는 사용하지 않음 (각 태스크별 자동 생성 hash 사용)
    -   **중요:**
        * instruction을 변경하면 prompt_hash도 변경되어 캐시를 찾을 수 없음
        * 캐시 생성과 학습 시 동일한 vlm_reuse_count 사용 필수

5.  **🔥🔥 VL 캐시 생성 트러블슈팅 및 최적화 (2025-11-18) 🔥🔥:**

    **⚠️ 문제 1: Tuple Detach 에러**
    -   **증상:** `'tuple' object has no attribute 'detach'` 에러 발생
    -   **원인:** `torch.split()`이 반환하는 tuple을 캐시 저장 시 직접 `.detach()` 호출
    -   **해결:** `models/vl_cache.py`의 `save_cache()` 메서드에 `flatten_to_tensor()` 재귀 함수 추가
        * 중첩된 tuple/list 구조를 재귀적으로 탐색
        * 각 텐서를 개별적으로 detach 및 CPU로 이동
        * 원본 데이터 구조(tuple/list/dict) 유지

    **⚠️ 문제 2: 비어있는 Image Features (torch.Size([1, 0, 2048]))**
    -   **증상:** VL 캐시의 image_features가 비어있음 (seq_len=0)
    -   **원인:** 잘못된 토큰 ID(151857)를 사용하여 비전 토큰 추출 시도
    -   **Qwen2.5-VL의 실제 토큰 구조:**
        * `<|vision_start|>` (151652): 비전 시퀀스 시작 마커
        * `<|image_pad|>` (151655): VLM이 실제 이미지 패치 임베딩으로 확장하는 플레이스홀더
        * `<|vision_end|>` (151653): 비전 시퀀스 종료 마커
    -   **해결:** `models/vl_encoder.py`의 이미지 토큰 추출 로직 완전 재작성
        * `<|vision_start|>`와 `<|vision_end|>` 위치 찾기
        * 두 마커 **사이의 모든** hidden states 추출 (특정 토큰 ID가 아님!)
        * 결과: 1320개의 이미지 패치 성공 추출 (264 patches × 5 views)

    **⚠️ 문제 3: 손상된 캐시 파일로 인한 Concatenation 에러**
    -   **증상:** 텐서 크기 불일치 (`[1, 0, 2048]` vs `[1, 1320, 2048]`)
    -   **원인:** 비전 토큰 추출 버그 수정 전에 생성된 invalid 캐시가 남아있음
    -   **해결:** `clean_invalid_cache.py` 유틸리티 생성
        * 모든 `.pt` 캐시 파일 검사
        * `image_features.shape[1] == 0`인 파일만 선택적 삭제
        * Dry-run 모드로 먼저 확인 후 삭제 가능

    **✅ 최종 최적화: 학습 전 완전 캐싱 전략 (TOTAL_TRAIN.sh)**
    -   **기존 문제:** 학습 중 cache backfill로 인한 속도 저하 및 GPU 메모리 낭비
    -   **해결책:**
        * **STEP 2.5:** Invalid 캐시 검사 및 정리 (사용자 확인 후 삭제)
        * **STEP 3:** 학습 전 100% 캐시 생성 (`--mode cache`, `vlm_reuse_count=1`)
        * **STEP 4:** 완전한 캐시로 빠른 학습 (`--use_cache --filter_by_cache --freeze_encoders`)
    -   **장점:**
        * 학습 중 VLM 로드 불필요 → 메모리 절약
        * Cache miss 없음 → 안정적이고 빠른 학습
        * 한 번 생성한 캐시는 여러 학습 실행에서 재사용 가능

    **🚨 중요 체크리스트:**
    1. ✅ `models/vl_cache.py`에서 `flatten_to_tensor()` 구현 확인
    2. ✅ `models/vl_encoder.py`에서 비전 토큰을 마커 사이 전체 시퀀스로 추출
    3. ✅ 학습 전 `clean_invalid_cache.py`로 손상된 캐시 제거
    4. ✅ `TOTAL_TRAIN.sh`에서 STEP 3으로 완전 캐싱 후 STEP 4로 학습
    5. ✅ 캐시 생성과 학습 시 동일한 `vlm_reuse_count` 사용 (일반적으로 1 또는 3)

---
Original Docstring:
Flow Matching VLA Training Script with Sensor Integration

Specialized training script for flow matching-based action prediction.
Based on Pi0 paper: https://arxiv.org/pdf/2410.24164v1

Usage:
    # Build cache first
    torchrun --nproc_per_node=4 TRAIN_FlowMatching.py --mode cache

    # Then train
    torchrun --nproc_per_node=4 TRAIN_FlowMatching.py --mode train
"""

from pydantic import PydanticDeprecatedSince20
import warnings
warnings.filterwarnings("ignore", message=".*cudnnException.*")
warnings.filterwarnings("ignore", message=".*Deterministic behavior.*")
warnings.filterwarnings("ignore", message=".*Flash Attention.*")

import argparse
import wandb
import io, shutil, threading, queue, time
import os
import sys
import re
import math
import glob
import pickle
import atexit
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from contextlib import nullcontext

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset, DataLoader, random_split
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler, Subset
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam

# Set seeds
import random
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ✅ OPTIMIZATION: Enable cudnn.benchmark for faster training (non-deterministic)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.use_deterministic_algorithms(False, warn_only=True)
torch.set_float32_matmul_precision("high")

# Import unified models and datasets
from models.unified_model import QwenVLAUnified
from vla_datasets.unified_dataset import (
    UnifiedVLADataset,
    create_unified_dataloader,
    unified_collate_fn,
)

# Import cache builder
import importlib.util
cache_module_path = Path(__file__).parent / "Make_VL_cache.py"
spec = importlib.util.spec_from_file_location("Make_VL_cache", cache_module_path)
cache_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache_module)
build_vl_cache_distributed_optimized = cache_module.build_vl_cache_distributed_optimized

# ======== I/O & Checkpoint Utils ========
STAGING_DIR = Path("/home/najo/NAS/VLA/tmp_stage")
CKPT_DIR = Path("./checkpoints")
STAGING_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

def _atomic_move(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.move(src, tmp)
    os.replace(tmp, dst)

def copy_to_local_then_load(src_path: Path, map_location):
    """네트워크 파일을 로컬 스테이징으로 빠르게 복사 후 torch.load"""
    if not src_path.exists():
        raise FileNotFoundError(str(src_path))
    local_copy = STAGING_DIR / src_path.name
    shutil.copy2(src_path, local_copy)
    try:
        return torch.load(local_copy, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(local_copy, map_location=map_location)

class AsyncCheckpointWriter:
    """학습은 그대로 진행, 저장은 백그라운드 스레드가 처리"""
    def __init__(self, max_queue=2, sync_every=0):
        self.q = queue.Queue(maxsize=max_queue)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.stop = False
        self.sync_every = sync_every
        self.thread.start()

    def _worker(self):
        last_sync = time.time()
        while not self.stop:
            try:
                payload = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            state_dict, final_dst = payload["state"], Path(payload["dst"])
            local_tmp = STAGING_DIR / (final_dst.name + f".{int(time.time())}.pt")
            torch.save(state_dict, local_tmp, _use_new_zipfile_serialization=True)
            if self.sync_every > 0 and (time.time() - last_sync) < self.sync_every:
                continue
            _atomic_move(local_tmp, final_dst)
            last_sync = time.time()

    def submit(self, state_dict, final_dst: Path):
        if self.q.full():
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
        self.q.put({"state": state_dict, "dst": str(final_dst)})

    def close(self):
        self.stop = True
        self.thread.join(timeout=5)

def build_trapezoid_scheduler(
    optimizer,
    total_steps: int,
    *,
    base_lr: float = 1e-4,
    min_lr: float = 1e-6,
    warmup_ratio: float = 0.03,
    hold_ratio: float = 0.02,
):
    """LLM 스타일: Warmup -> Hold -> Cosine Decay"""
    warmup_steps = int(total_steps * warmup_ratio)
    hold_steps = int(total_steps * hold_ratio)
    decay_steps = max(1, total_steps - warmup_steps - hold_steps)
    floor = min_lr / max(base_lr, 1e-12)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        elif step < warmup_steps + hold_steps:
            return 1.0
        else:
            t = (step - warmup_steps - hold_steps) / decay_steps
            cos_val = 0.5 * (1.0 + math.cos(math.pi * t))
            return floor + (1.0 - floor) * cos_val

    sched = LambdaLR(optimizer, lr_lambda=lr_lambda)
    prev_lr = base_lr * lr_lambda(0)
    for g in optimizer.param_groups:
        g["lr"] = prev_lr
    return sched


def _count_cache_hits(vl_cache_batch):
    """Return (#hits, #samples) for a batch-level vl_cache list."""
    if not isinstance(vl_cache_batch, (list, tuple)):
        return 0, 0

    hits = 0
    total = len(vl_cache_batch)

    for entry in vl_cache_batch:
        if entry is None:
            continue
        if isinstance(entry, torch.Tensor):
            if entry.numel() > 0:
                hits += 1
        elif isinstance(entry, (list, tuple)):
            if any(isinstance(x, torch.Tensor) and x.numel() > 0 for x in entry):
                hits += 1
        else:
            numel = getattr(entry, "numel", None)
            if callable(numel):
                try:
                    if entry.numel() > 0:
                        hits += 1
                except Exception:
                    continue

    return hits, total


def _sync_cache_stats(hits, total, device):
    """All-reduce cache hit stats across DDP ranks."""
    if dist.is_available() and dist.is_initialized():
        stats = torch.tensor([hits, total], dtype=torch.float64, device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        return stats[0].item(), stats[1].item()
    return float(hits), float(total)


def validate_cache_file(cache_path: Path, verbose: bool = False):
    """
    Validate a single cache file.
    Returns: (is_valid, error_message, cache_info)
    """
    try:
        cache_data = torch.load(cache_path, map_location='cpu')

        # Check if cache has required keys
        required_keys = ['image_features', 'text_features']
        missing_keys = [k for k in required_keys if k not in cache_data]
        if missing_keys:
            return False, f"Missing keys: {missing_keys}", None

        # Check image_features
        image_features = cache_data['image_features']
        if isinstance(image_features, (list, tuple)):
            # Handle tuple/list of tensors
            total_elements = sum(f.numel() if isinstance(f, torch.Tensor) else 0 for f in image_features)
            if total_elements == 0:
                return False, "Empty image_features (all tensors have 0 elements)", None
            shapes = [f.shape if isinstance(f, torch.Tensor) else None for f in image_features]
        elif isinstance(image_features, torch.Tensor):
            if image_features.numel() == 0 or image_features.shape[1] == 0:
                return False, f"Empty image_features tensor: {image_features.shape}", None
            shapes = [image_features.shape]
        else:
            return False, f"Invalid image_features type: {type(image_features)}", None

        cache_info = {
            'file': cache_path.name,
            'image_features_shapes': shapes,
            'text_features_shape': cache_data['text_features'].shape if isinstance(cache_data['text_features'], torch.Tensor) else None,
            'size_mb': cache_path.stat().st_size / 1024 / 1024,
        }

        if verbose:
            print(f"✅ Valid: {cache_path.name}")
            print(f"   Image features: {shapes}")
            print(f"   Text features: {cache_info['text_features_shape']}")
            print(f"   Size: {cache_info['size_mb']:.2f} MB")

        return True, None, cache_info

    except Exception as e:
        return False, f"Load error: {str(e)}", None


def validate_all_caches(cache_root: Path, rank: int = 0, verbose: bool = True):
    """
    Validate all cache files in cache_root directory.
    Returns summary statistics.
    """
    if rank != 0:
        return None

    if not cache_root.exists():
        print(f"❌ Cache directory does not exist: {cache_root}")
        return None

    print(f"\n{'='*80}")
    print(f"🔍 CACHE VALIDATION REPORT")
    print(f"{'='*80}")
    print(f"Cache root: {cache_root}")

    # Find all .pt files
    cache_files = list(cache_root.rglob("*.pt"))

    if not cache_files:
        print(f"⚠️ No cache files found in {cache_root}")
        return None

    print(f"Found {len(cache_files)} cache files\n")

    valid_files = []
    invalid_files = []
    total_size_mb = 0

    # Group by prompt_hash directory
    by_prompt_hash = {}
    for cache_file in cache_files:
        prompt_hash = cache_file.parent.name
        if prompt_hash not in by_prompt_hash:
            by_prompt_hash[prompt_hash] = []
        by_prompt_hash[prompt_hash].append(cache_file)

    print(f"📁 Cache organized by {len(by_prompt_hash)} prompt_hash directories:")
    for prompt_hash, files in by_prompt_hash.items():
        print(f"   {prompt_hash}: {len(files)} files")
    print()

    # Validate each file
    for cache_file in tqdm(cache_files, desc="Validating caches", disable=(not verbose)):
        is_valid, error_msg, cache_info = validate_cache_file(cache_file, verbose=False)

        if is_valid:
            valid_files.append(cache_file)
            total_size_mb += cache_info['size_mb']
        else:
            invalid_files.append((cache_file, error_msg))
            if verbose:
                print(f"❌ Invalid: {cache_file.relative_to(cache_root)}")
                print(f"   Error: {error_msg}")

    # Print summary
    print(f"\n{'='*80}")
    print(f"📊 VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total files:        {len(cache_files)}")
    print(f"✅ Valid files:     {len(valid_files)} ({len(valid_files)/len(cache_files)*100:.1f}%)")
    print(f"❌ Invalid files:   {len(invalid_files)} ({len(invalid_files)/len(cache_files)*100:.1f}%)")
    print(f"💾 Total size:      {total_size_mb:.2f} MB")
    print(f"{'='*80}\n")

    if invalid_files:
        print(f"⚠️ Invalid cache files found:")
        for cache_file, error_msg in invalid_files[:10]:  # Show first 10
            print(f"   {cache_file.relative_to(cache_root)}: {error_msg}")
        if len(invalid_files) > 10:
            print(f"   ... and {len(invalid_files) - 10} more")
        print()

    return {
        'total': len(cache_files),
        'valid': len(valid_files),
        'invalid': len(invalid_files),
        'total_size_mb': total_size_mb,
        'invalid_files': invalid_files,
    }

# ===========================================================
# 초기화
# ===========================================================
def setup_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # torchrun에서 LOCAL_RANK는 프로세스별 GPU ID임
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    print(f"[Rank {rank}] using device {device}")
    return rank, world_size, local_rank, device

# ============================================================
# Unified Dataloader Builder
# ============================================================
def build_dataloaders(args, rank, world_size, use_cache=True, cache_build_only=False):
    """
    Build unified dataloaders combining:
      ① Old format datasets
      ② New format datasets
    """
    if rank == 0:
        print(f"[RANK {rank}] 🚀 Building Unified Async Dataloaders (world_size={world_size})")

    # Build TRAIN dataloader
    print("\n📦 Creating TRAIN dataloader (weighted mix of old/new)...")

    train_loader = create_unified_dataloader(
        new_dataset_paths=args.dataset_paths,
        dataset_weights=args.dataset_weights,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        horizon=args.horizon if hasattr(args, "horizon") else 8,
        vlm_reuse_count=args.vlm_reuse_count,
        sensor_window_size=args.sensor_window_size if hasattr(args, "sensor_window_size") else 65,
        robot_window_size=getattr(args, "robot_window_size", 100),
        action_expert_hz=args.action_expert_hz if hasattr(args, "action_expert_hz") else 10,
        distributed=True,
        rank=rank,
        world_size=world_size,
        use_cache=use_cache,  # Pass the flag here
        use_augmentation=getattr(args, "use_augmentation", False),
        augmentation_prob=getattr(args, "augmentation_prob", 0.10),
        cache_build_only=cache_build_only,
        cache_root=getattr(args, "cache_root", "/home/najo/NAS/VLA/dataset/cache/qwen_vl_features"),
        prompt_hash_override=getattr(args, "prompt_hash_override", None),
        skip_dataset_stats=getattr(args, "skip_dataset_stats", False),
        filter_by_cache=getattr(args, "filter_by_cache", False),
    )

    if cache_build_only:
        return train_loader, None

    # Build VAL dataloader
    print("\n📦 Creating VAL dataloader (validation subset)...")

    val_dataset_path = args.dataset_paths[0] if args.dataset_paths else None
    val_datasets = []
    if val_dataset_path:
        try:
            new_path = Path(val_dataset_path)
            task_dirs = [d for d in new_path.iterdir() if d.is_dir()]
            picked = []
            for t in task_dirs:
                ep = next((d for d in t.iterdir() if d.is_dir() and (d.name.startswith('episode_') or d.name.startswith('data_collection_'))), None)
                if ep:
                    picked.append(ep)
            for ep_dir in picked:
                ds = UnifiedVLADataset(
                    data_dir=str(ep_dir), 
                    format='new',
                    horizon=args.horizon if hasattr(args, "horizon") else 8,
                    vlm_reuse_count=args.vlm_reuse_count if hasattr(args, "vlm_reuse_count") else 3,
                    sensor_window_size=args.sensor_window_size if hasattr(args, "sensor_window_size") else 65,
                    action_expert_hz=args.action_expert_hz if hasattr(args, "action_expert_hz") else 10,
                    cache_root=getattr(args, "cache_root", "/home/najo/NAS/VLA/dataset/cache/qwen_vl_features"),
                    prompt_hash_override=getattr(args, "prompt_hash_override", None),
                    filter_by_cache=getattr(args, "filter_by_cache", False),
                )
                val_datasets.append(ds)
        except (StopIteration, FileNotFoundError) as e:
            print(f"⚠️ Could not create validation set from {val_dataset_path}: {e}")

    from torch.utils.data import ConcatDataset
    if len(val_datasets) == 0:
        print("⚠️ No validation datasets found, using train subset instead.")
        if hasattr(train_loader.dataset, 'datasets'):
             val_datasets = [next(iter(train_loader.dataset.datasets))]
        else:
            val_dataset_length = len(train_loader.dataset)
            val_indices = list(range(int(val_dataset_length * 0.1)))
            val_datasets = [Subset(train_loader.dataset, val_indices)]

    val_dataset = ConcatDataset(val_datasets)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=val_sampler,
        collate_fn=unified_collate_fn,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    if rank == 0:
        print(f"✅ TRAIN loader: {len(train_loader)} batches | VAL loader: {len(val_loader)} batches")

    return train_loader, val_loader

# ===========================================================
# Flow Matching 학습 루프
# ===========================================================
def Train_FlowMatching(
    model_engine,
    optimizer,
    scheduler,
    data_loader,
    num_epochs=3,
    grad_accum_steps=8,
    device="cuda",
    val_loader=None,
    start_epoch=0,
    sensor_enabled=True,
    sensor_loss_weight=2.0,
    finetune_vl='none',
):
    """Flow Matching training loop - DeepSpeed OPTIMIZED"""
    rank = dist.get_rank()
    writer = AsyncCheckpointWriter(max_queue=2, sync_every=0) if rank == 0 else None

    model_engine.train()
    if rank == 0:
        wandb.init(
            project="QwenVLA-FlowMatching",
            name=f"flow_matching_ds_{time.strftime('%m%d_%H%M')}",
            resume="allow",
            id=f"qvla_flow_matching_ds_{int(time.time())}",
            settings=wandb.Settings(start_method="thread", _disable_stats=True),
            config={
                "model_type": "flow_matching",
                "lr": model_engine.get_lr()[0] if isinstance(model_engine, deepspeed.DeepSpeedEngine) else optimizer.param_groups[0]['lr'],
                "grad_accum_steps": grad_accum_steps,
                "epochs": num_epochs,
                "deepspeed": isinstance(model_engine, deepspeed.DeepSpeedEngine),
                "sensor_enabled": sensor_enabled,
                "sensor_loss_weight": sensor_loss_weight,
            }
        )

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(start_epoch, start_epoch + num_epochs):
        if isinstance(data_loader.sampler, DistributedSampler):
            data_loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        total_sensor_samples = 0
        total_nonsensor_samples = 0
        epoch_cache_hits = 0
        epoch_cache_total = 0

        model_engine.train()

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"📚 Epoch {epoch+1}/{start_epoch + num_epochs}")
            print(f"{'='*60}")

        pbar = tqdm(enumerate(data_loader), total=len(data_loader),
                    desc=f"[Rank {rank}] Epoch {epoch+1}",
                    disable=(rank != 0))

        for step, batch in pbar:
            try:
                instructions = batch["instruction"]
                image_inputs = batch["images"]
                gt_actions = batch["actions"].to(device, dtype=torch.bfloat16, non_blocking=True)
                vl_cache_batch = batch.get("vl_cache")
                hits, total = _count_cache_hits(vl_cache_batch)
                epoch_cache_hits += hits
                epoch_cache_total += total

                # ✅ FLOW MATCHING: Use all actions (B, 8, 7)
                sensor_data = (
                    batch["sensor_data"].to(device, dtype=torch.bfloat16, non_blocking=True)
                    if sensor_enabled else None
                )
                has_sensor_mask = (
                    batch["has_sensor_mask"].to(device, non_blocking=True)
                    if sensor_enabled else None
                )

                # Robot states
                robot_states = None
                if "robot_states" in batch and sensor_enabled:
                    try:
                        robot_states = batch["robot_states"].to(device, non_blocking=True)
                    except Exception as e:
                        if rank == 0 and step == 0:
                            print(f"⚠️ Failed to load robot_states: {e}")
                        robot_states = None

                cache_metadata = {
                    "dataset_names": batch.get("episode_ids"),
                    "vlm_indices": batch.get("vlm_indices"),
                    "prompt_hashes": batch.get("prompt_hash"),
                }

                # Flow matching: model returns loss directly
                flow_loss, _, _ = model_engine(
                    text_inputs=instructions,
                    image_inputs=image_inputs,
                    actions=gt_actions,  # ✅ Use 'actions' parameter
                    cache_keys=batch["cache_keys"],
                    sensor_data=sensor_data if sensor_enabled else None,
                    robot_states=robot_states,
                    vl_cache_tokens=batch.get("vl_cache"),
                    vl_cache_metadata=cache_metadata,
                )

                # ✅ Count sensor samples for logging
                if sensor_enabled and has_sensor_mask is not None:
                    total_sensor_samples += has_sensor_mask.sum().item()
                    total_nonsensor_samples += (~has_sensor_mask).sum().item()

                # DeepSpeed handles gradient accumulation internally
                if isinstance(model_engine, deepspeed.DeepSpeedEngine):
                    model_engine.backward(flow_loss)
                    model_engine.step()
                else:
                    # Standard PyTorch training
                    flow_loss.backward()
                    if (step + 1) % grad_accum_steps == 0:
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()


                total_loss += flow_loss.item()
                global_step += 1

                if rank == 0:
                    lr = model_engine.get_lr()[0] if isinstance(model_engine, deepspeed.DeepSpeedEngine) else optimizer.param_groups[0]['lr']
                    running_cache_ratio = (
                        epoch_cache_hits / epoch_cache_total
                        if epoch_cache_total > 0 else 0.0
                    )
                    cache_status = (
                        f"{running_cache_ratio*100:.1f}% ({epoch_cache_hits}/{epoch_cache_total})"
                        if epoch_cache_total > 0 else "0.0% (0/0)"
                    )
                    postfix_dict = {
                        "loss": f"{flow_loss.item():.6f}",
                        "lr": f"{lr:.2e}",
                        "cache": cache_status,
                    }
                    if sensor_enabled:
                        postfix_dict["sensor"] = f"{total_sensor_samples}/{total_sensor_samples+total_nonsensor_samples}"
                    pbar.set_postfix(postfix_dict)

                    log_dict = {
                        "train/loss_step": flow_loss.item(),
                        "train/lr": lr,
                        "global_step": global_step,
                        "train/cache_hit_ratio_step": running_cache_ratio,
                    }
                    if sensor_enabled:
                        log_dict["train/sensor_samples"] = total_sensor_samples
                        log_dict["train/nonsensor_samples"] = total_nonsensor_samples
                    wandb.log(log_dict)

            except FileNotFoundError as e:
                if rank == 0:
                    pbar.write(f"⚠️ [Rank {rank}] 캐시 파일 없음, Batch {step} 스킵. (오류: {e})")
                continue

        avg_loss_tensor = torch.tensor(total_loss / len(data_loader), device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = avg_loss_tensor.item()

        # Validation
        val_loss = None
        if val_loader is not None:
            model_engine.eval()
            val_loss_sum, val_count = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    try:
                        gt_actions = batch["actions"].to(device, dtype=torch.bfloat16, non_blocking=True)

                        sensor_data = (
                            batch["sensor_data"].to(device, dtype=torch.bfloat16, non_blocking=True)
                            if sensor_enabled else None
                        )
                        robot_states = (
                            batch["robot_states"].to(device, non_blocking=True)
                            if "robot_states" in batch and sensor_enabled else None
                        )

                        cache_metadata = {
                            "dataset_names": batch.get("episode_ids"),
                            "vlm_indices": batch.get("vlm_indices"),
                            "prompt_hashes": batch.get("prompt_hash"),
                        }

                        # Flow matching: model returns loss directly
                        loss, _, _ = model_engine(
                            text_inputs=batch["instruction"],
                            image_inputs=batch["images"],
                            actions=gt_actions,
                            cache_keys=batch["cache_keys"],
                            sensor_data=sensor_data if sensor_enabled else None,
                            robot_states=robot_states,
                            vl_cache_tokens=batch.get("vl_cache"),
                            vl_cache_metadata=cache_metadata,
                        )
                        if loss.ndim > 0:
                            loss = loss.mean()
                        val_loss_sum += loss.item()
                        val_count += 1
                    except FileNotFoundError:
                        if rank == 0:
                            print(f"⚠️ [Rank {rank}] Validation 중 캐시 파일 없음, 스킵.")
                        continue

            val_loss = val_loss_sum / max(1, val_count)
            model_engine.train()

        synced_hits, synced_total = _sync_cache_stats(epoch_cache_hits, epoch_cache_total, device)
        cache_hit_ratio = (synced_hits / synced_total) if synced_total > 0 else 0.0

        # Checkpoint saving
        if rank == 0:
            import psutil, gc
            model_to_save = model_engine.module if isinstance(model_engine, deepspeed.DeepSpeedEngine) else model_engine
            trainable = sum(p.numel() for p in model_to_save.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model_to_save.parameters())
            frozen = total_params - trainable

            gpu_mem = torch.cuda.memory_allocated()/1e9
            cpu_mem = psutil.virtual_memory().percent
            gc.collect()

            lr = model_engine.get_lr()[0] if isinstance(model_engine, deepspeed.DeepSpeedEngine) else optimizer.param_groups[0]['lr']
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            log_dict = {
                "epoch": epoch + 1,
                "train/loss_epoch": avg_loss,
                "val/loss_epoch": val_loss if val_loss else None,
                "params/trainable_M": trainable / 1e6,
                "params/frozen_M": frozen / 1e6,
                "params/frozen_ratio": frozen / total_params,
                "system/gpu_mem_GB": gpu_mem,
                "system/cpu_mem_%": cpu_mem,
                "lr/base_lr": lr,
                "train/cache_hit_ratio": cache_hit_ratio,
                "train/processed_samples_epoch": synced_total,
                "dataset/total_samples": len(data_loader.dataset),
                "dataset/total_batches": len(data_loader) * world_size,
            }

            if sensor_enabled:
                log_dict["train/epoch_sensor_samples"] = total_sensor_samples
                log_dict["train/epoch_nonsensor_samples"] = total_nonsensor_samples
                log_dict["train/sensor_ratio"] = total_sensor_samples / max(1, total_sensor_samples + total_nonsensor_samples)

            wandb.log(log_dict)
            print(f"\n📊 Epoch {epoch+1} Summary | Train: {avg_loss:.8f} | " +
                  (f"Val: {val_loss:.8f}" if val_loss else ""))
            print(f"   Cache hit ratio: {cache_hit_ratio*100:.2f}% ({int(synced_hits)}/{int(synced_total)})")

            # Simple .pt checkpoint saving (model weights only)
            is_best = val_loss is not None and val_loss < best_val_loss

            if is_best:
                best_val_loss = val_loss
                checkpoint_name = "flow_matching_best.pt"
                print(f"🏆 [Best] Validation improved → saving checkpoint")
            else:
                checkpoint_name = "flow_matching_latest.pt"
                print(f"💾 Saving latest checkpoint")

            # Save only model weights (no optimizer, no scheduler)
            checkpoint_path = CKPT_DIR / checkpoint_name
            ckpt_data = {
                "epoch": epoch + 1,
                "model_state_dict": model_to_save.state_dict(),
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "finetune_vl": finetune_vl,
            }
            torch.save(ckpt_data, checkpoint_path)
            print(f"✅ Checkpoint saved: {checkpoint_path}")

    if rank == 0 and writer is not None:
        atexit.register(writer.close)

    if rank == 0:
        wandb.finish()

# ===========================================================
# Main
# ===========================================================
def main():
    parser = argparse.ArgumentParser(description='Flow Matching VLA Training with Sensor')

    # Mode (for cache building)
    parser.add_argument('--mode', type=str, choices=['cache', 'train'], default='train',
                        help='Mode: cache (build VL cache) or train')

    # Dataset
    parser.add_argument('--dataset_paths', type=str, nargs='+', required=True,
                        help='List of paths to the datasets.')
    parser.add_argument('--dataset_weights', type=str, nargs='+',
                        help='List of weights for each dataset, in path:weight format.')
    parser.add_argument('--cache_root', type=str,
                        default='/home/najo/NAS/VLA/dataset/cache/qwen_vl_features',
                        help='VL feature cache directory (prompt-hash aware).')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--sensor_lr', type=float, default=5e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument('--hold_ratio', type=float, default=0.02)
    parser.add_argument('--sched_on', type=str, choices=['step', 'epoch'], default='step')

    # Sensor options
    parser.add_argument('--sensor_enabled', action='store_true', default=True,
                        help='Enable sensor encoder training')
    parser.add_argument('--sensor_hidden_dim', type=int, default=512,
                        help='Conv backbone initial channel (512=heavy final 4096, 256=light final 2048)')
    parser.add_argument('--sensor_transformer_dim', type=int, default=None,
                        help='Transformer dimension for projection (None=auto from conv, 1024=medium/lightweight)')
    parser.add_argument('--sensor_loss_weight', type=float, default=2.0)
    parser.add_argument('--fusion_strategy', type=str, default='concat',
                        choices=['concat', 'cross_attention', 'gated'])

    # Image resize
    parser.add_argument('--image_resize_height', type=int, default=360)
    parser.add_argument('--image_resize_width', type=int, default=640)

    # Pre-trained encoder loading
    parser.add_argument('--load_sensor_encoder_checkpoint', type=str, default='./checkpoints/sensor_clip_best.pth',
                        help='Path to pre-trained sensor encoder checkpoint.')
    parser.add_argument('--load_robot_state_encoder_checkpoint', type=str, default='./checkpoints/robot_state_mae_best.pth',
                        help='Path to pre-trained robot state encoder checkpoint.')
    parser.add_argument('--freeze_encoders', action='store_true', help='Freeze both sensor and robot state encoders after loading weights.')
    parser.add_argument('--freeze_sensor_encoder', action='store_true', help='Freeze only the sensor encoder.')
    parser.add_argument('--freeze_robot_state_encoder', action='store_true', help='Freeze only the robot state encoder.')

    # Data augmentation
    parser.add_argument('--use_augmentation', action='store_true', help='Enable minimal image augmentation (only works without cache)')
    parser.add_argument('--augmentation_prob', type=float, default=0.10, help='Augmentation probability (default: 0.10)')

    # Cache management
    parser.add_argument('--prompt_hash_override', type=str, default=None,
                        help='Override prompt hash for cache lookup (use when cache was built with different instruction)')
    parser.add_argument('--filter_by_cache', action='store_true',
                        help='Only use samples that have VL cache available (skip samples without cache)')

    # Other
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--use_cache', action='store_true', help='Enable VL feature caching')
    parser.add_argument('--finetune_vl', type=str, default='none', choices=['none', 'lora', 'full'], help='Fine-tuning mode for VL model')
    parser.add_argument('--vlm_reuse_count', type=int, default=3, help='Number of frames to share a single VLM feature. Set to 1 for 100%% cache generation.')
    parser.add_argument('--cache_loader_only', action='store_true', help='Use lightweight dataloader optimized for cache building')
    parser.add_argument('--skip_dataset_stats', action='store_true', help='Skip dataset statistics collection and printing (faster startup)')
    parser.add_argument('--debug_mode', action='store_true', help='Enable verbose VL/cache debug logging')

    # DeepSpeed args
    parser.add_argument('--deepspeed_config', type=str, default='configs/deepspeed_zero2.json',
                        help='Path to DeepSpeed config file')
    parser.add_argument('--action_expert_hidden_dim', type=int, default=1024, help='Hidden dimension of the action expert.')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training (set by DeepSpeed)')

    args = parser.parse_args()

    # Setup distributed
    rank, world_size, local_rank, device = setup_distributed()
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"🚀 Flow Matching VLA Training")
        print(f"   Mode: {args.mode.upper()}")
        print(f"   World Size: {world_size}")

    vl_model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    cache_dir = Path(args.cache_root)
    cache_loader_mode = args.cache_loader_only or (args.mode == 'cache')

    # Cache build mode
    if args.mode == 'cache':
        if rank == 0:
            print("\n" + "="*80)
            print("🏗️  STARTING VL CACHE BUILDING")
            print("="*80)
            print(f"Cache root: {cache_dir}")
            print(f"VLM reuse count: {args.vlm_reuse_count}")
            print(f"Batch size: {args.batch_size}")
            print(f"Num workers: {args.num_workers}")
            print(f"Debug mode: {args.debug_mode}")
            print("="*80 + "\n")

        # Count existing caches before building
        if rank == 0:
            existing_caches = list(cache_dir.rglob("*.pt")) if cache_dir.exists() else []
            print(f"📦 Existing cache files: {len(existing_caches)}\n")

        # 1. Dataloader (use_cache=False is crucial)
        train_loader, _ = build_dataloaders(
            args,
            rank,
            world_size,
            use_cache=True,  # 캐시 생성 시점과 학습 시점의 프롬프트/해시를 일치시켜 재사용 가능하게 함
            cache_build_only=cache_loader_mode,
        )

        # 2. Model (needed for its processor and VL model)
        if rank == 0:
            print("⏳ Loading VL model for cache generation...")

        model = QwenVLAUnified(
            model_type='flow_matching', vl_model_name=vl_model_name, action_dim=7, horizon=8,
            hidden_dim=1024, sensor_enabled=False, # Sensor data not needed for VL cache
            finetune_vl='none',
            image_resize_height=args.image_resize_height, image_resize_width=args.image_resize_width,
            device_map=None,
            external_cache_root=args.cache_root,
            debug_mode=args.debug_mode,
        )
        model = model.to(device)

        # Pass cache_dir to the model so the builder can find it
        # This is a bit of a hack, but it's how the cache builder is designed
        model.cache_dir = cache_dir

        if rank == 0:
            print("✅ VL model loaded\n")
            print(f"🚀 Starting cache generation for {len(train_loader.dataset)} samples...")
            print(f"   Expected VLM forward passes: ~{len(train_loader.dataset) // args.vlm_reuse_count}\n")

        # 3. Run the cache building process
        build_vl_cache_distributed_optimized(
            model=model,
            dataset=train_loader.dataset,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        # Ensure all processes sync up before validation
        if dist.is_initialized():
            dist.barrier()

        # 4. Validate generated caches
        if rank == 0:
            print("\n" + "="*80)
            print("✅ CACHE BUILDING COMPLETE")
            print("="*80)

            # Count new caches
            all_caches = list(cache_dir.rglob("*.pt")) if cache_dir.exists() else []
            new_caches = len(all_caches) - len(existing_caches)
            print(f"📦 Total cache files: {len(all_caches)} (+{new_caches} new)")
            print()

            # Validate all caches
            validation_results = validate_all_caches(
                cache_root=cache_dir,
                rank=rank,
                verbose=True
            )

            # Show recommendation
            if validation_results and validation_results['invalid'] > 0:
                print("⚠️  RECOMMENDATION:")
                print(f"   Run clean_invalid_cache.py to remove {validation_results['invalid']} invalid cache files")
                print(f"   Command: python clean_invalid_cache.py --cache_root {cache_dir}")
            elif validation_results:
                print("✨ All cache files are valid! Ready for training.")

            print("="*80 + "\n")

        # Ensure all processes sync up before exiting
        if dist.is_initialized():
            dist.barrier()

        return

    # Training mode
    if rank == 0:
        print("\n" + "="*80)
        print("🎓 STARTING TRAINING MODE")
        print("="*80)
        print(f"Cache root: {cache_dir}")
        print(f"Use cache: {args.use_cache}")
        print(f"Filter by cache: {args.filter_by_cache}")

        # Check cache status
        if args.use_cache:
            if cache_dir.exists():
                cache_files = list(cache_dir.rglob("*.pt"))
                print(f"📦 Found {len(cache_files)} cache files")

                # Run quick validation in debug mode
                if args.debug_mode:
                    print("\n🔍 Running cache validation (debug mode enabled)...")
                    validation_results = validate_all_caches(
                        cache_root=cache_dir,
                        rank=rank,
                        verbose=True
                    )
                    if validation_results and validation_results['invalid'] > 0:
                        print(f"\n⚠️  WARNING: Found {validation_results['invalid']} invalid cache files!")
                        print(f"   Consider running: python clean_invalid_cache.py --cache_root {cache_dir}\n")
            else:
                print(f"⚠️  WARNING: Cache directory does not exist!")
                print(f"   Run with --mode cache first to generate caches\n")
        else:
            print("⚠️  Cache disabled - VLM will run in real-time (slower)")

        print("="*80 + "\n")

    train_loader, val_loader = build_dataloaders(args, rank, world_size, use_cache=args.use_cache, cache_build_only=args.cache_loader_only)

    if rank == 0: print("⏳ Initializing model for training...")

    # Cache 사용 시 VLM 로드 스킵 (메모리 절약). filter_by_cache가 활성화되어 캐시 누락이 없을 때만 사용.
    use_cache_only_mode = args.use_cache and args.finetune_vl == 'none' and args.filter_by_cache

    model = QwenVLAUnified(
        model_type='flow_matching', vl_model_name=vl_model_name, action_dim=7, horizon=8,
        hidden_dim=1024, sensor_enabled=args.sensor_enabled,
        sensor_input_channels=1026,
        sensor_temporal_length=65,
        sensor_output_dim=512,
        robot_state_enabled=args.sensor_enabled,
        robot_state_output_dim=512, # Match pre-training architecture
        finetune_vl=args.finetune_vl,
        action_expert_hidden_dim=args.action_expert_hidden_dim,
        image_resize_height=args.image_resize_height, image_resize_width=args.image_resize_width,
        device_map=None,
        external_cache_root=args.cache_root,
        cache_only_mode=use_cache_only_mode,  # VLM freeze + cache 사용 시 VLM 로드 스킵
        debug_mode=args.debug_mode,
    )
    model = model.to(device)

    # Load pre-trained encoders on rank 0
    if rank == 0:
        # Load Sensor Encoder
        if args.sensor_enabled and args.load_sensor_encoder_checkpoint and os.path.exists(args.load_sensor_encoder_checkpoint):
            print(f"Loading SensorEncoder from: {args.load_sensor_encoder_checkpoint}")
            ckpt = torch.load(args.load_sensor_encoder_checkpoint, map_location='cpu')
            # Correctly strip the prefix from the unwrapped pre-trained model
            prefix = 'sensor_encoder.'
            sensor_encoder_state_dict = {k.replace(prefix, ''): v for k, v in ckpt['model_state_dict'].items() if k.startswith(prefix)}

            # Load with strict=False to handle any potential mismatches
            missing_keys, unexpected_keys = model.sensor_encoder.load_state_dict(sensor_encoder_state_dict, strict=False)
            if missing_keys:
                print(f"   ⚠️ Missing keys in SensorEncoder: {missing_keys}")
            if unexpected_keys:
                print(f"   ⚠️ Unexpected keys in SensorEncoder: {unexpected_keys}")
            print("✅ SensorEncoder weights loaded.")

        # Load Robot State Encoder
        if args.sensor_enabled and args.load_robot_state_encoder_checkpoint and os.path.exists(args.load_robot_state_encoder_checkpoint):
            print(f"Loading RobotStateEncoder from: {args.load_robot_state_encoder_checkpoint}")
            ckpt = torch.load(args.load_robot_state_encoder_checkpoint, map_location='cpu')
            prefix = 'encoder.'
            robot_state_encoder_state_dict = {k.replace(prefix, ''): v for k, v in ckpt['model_state_dict'].items() if k.startswith(prefix)}

            # Fix key mismatch: transformer_layers.* -> transformer_encoder.layers.*
            fixed_state_dict = {}
            for k, v in robot_state_encoder_state_dict.items():
                if k.startswith('transformer_layers.'):
                    # Replace transformer_layers with transformer_encoder.layers
                    new_key = k.replace('transformer_layers.', 'transformer_encoder.layers.')
                    fixed_state_dict[new_key] = v
                else:
                    fixed_state_dict[k] = v
            robot_state_encoder_state_dict = fixed_state_dict

            # Handle positional encoding size mismatch (65 -> 100)
            if 'pos_encoder' in robot_state_encoder_state_dict:
                pretrained_pos_enc = robot_state_encoder_state_dict['pos_encoder']  # [1, 65, 256]
                current_pos_enc = model.robot_state_encoder.pos_encoder  # [1, 100, 256]

                if pretrained_pos_enc.shape[1] != current_pos_enc.shape[1]:
                    print(f"   ⚠️ Positional encoding size mismatch: {pretrained_pos_enc.shape} -> {current_pos_enc.shape}")
                    print(f"   🔧 Interpolating positional encoding from {pretrained_pos_enc.shape[1]} to {current_pos_enc.shape[1]}")

                    # Interpolate along the sequence dimension
                    pretrained_pos_enc = pretrained_pos_enc.permute(0, 2, 1)  # [1, 256, 65]
                    interpolated = torch.nn.functional.interpolate(
                        pretrained_pos_enc,
                        size=current_pos_enc.shape[1],
                        mode='linear',
                        align_corners=True
                    )
                    robot_state_encoder_state_dict['pos_encoder'] = interpolated.permute(0, 2, 1)  # [1, 100, 256]

            # Load with strict=False to handle any potential mismatches
            missing_keys, unexpected_keys = model.robot_state_encoder.load_state_dict(robot_state_encoder_state_dict, strict=False)
            if missing_keys:
                print(f"   ⚠️ Missing keys in RobotStateEncoder: {missing_keys}")
            if unexpected_keys:
                print(f"   ⚠️ Unexpected keys in RobotStateEncoder: {unexpected_keys}")
            print("✅ RobotStateEncoder weights loaded.")

    # Determine per-encoder freeze configuration
    freeze_sensor_encoder = args.freeze_encoders or args.freeze_sensor_encoder
    freeze_robot_encoder = args.freeze_encoders or args.freeze_robot_state_encoder

    # Freeze encoders BEFORE DeepSpeed initialization
    if freeze_sensor_encoder or freeze_robot_encoder:
        if rank == 0:
            msg = []
            if freeze_sensor_encoder:
                msg.append("Sensor")
            if freeze_robot_encoder:
                msg.append("RobotState")
            print(f"🧊 Freezing {' & '.join(msg)} Encoder(s)...")

        if freeze_sensor_encoder:
            for param in model.sensor_encoder.parameters():
                param.requires_grad = False
        if freeze_robot_encoder:
            for param in model.robot_state_encoder.parameters():
                param.requires_grad = False

        if rank == 0:
            print("✅ Selected encoders frozen.")

    # When doing LoRA fine-tuning, freeze everything except VLM (Action Expert included)
    if args.finetune_vl == 'lora':
        if rank == 0: print("🧊 LoRA mode: Freezing Action Expert (only VLM LoRA adapters will be trained)...")
        for param in model.action_expert.parameters():
            param.requires_grad = False
        # Also freeze sensor encoders if not already frozen
        if not freeze_sensor_encoder:
            for param in model.sensor_encoder.parameters():
                param.requires_grad = False
            freeze_sensor_encoder = True
        if not freeze_robot_encoder:
            for param in model.robot_state_encoder.parameters():
                param.requires_grad = False
            freeze_robot_encoder = True
        if rank == 0: print("✅ Action Expert frozen. Only VLM LoRA adapters are trainable.")

    # Sync all processes before deepspeed.initialize() to prevent timeouts
    # Rank 0 might be busy loading checkpoints, while other ranks wait.
    if dist.is_initialized():
        dist.barrier()

    # Initialize model, optimizer, and scheduler
    if use_cache_only_mode:
        if rank == 0: print("⚡ Cache-only mode: Using standard PyTorch initialization (DeepSpeed skipped)")
        # Standard PyTorch initialization
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # Calculate scheduler parameters
        total_steps = max(1, (len(train_loader) * args.epochs) // max(1, args.grad_accum))
        warmup_steps = int(total_steps * args.warmup_ratio)

        scheduler = build_trapezoid_scheduler(
            optimizer,
            total_steps=total_steps,
            base_lr=args.lr,
            min_lr=args.min_lr,
            warmup_ratio=args.warmup_ratio,
            hold_ratio=args.hold_ratio,
        )
        model_engine = model # In non-deepspeed mode, model_engine is just the model
    else:
        if rank == 0: print("🚀 Full training mode: Initializing with DeepSpeed")
        # Calculate scheduler parameters for DeepSpeed config
        total_steps = max(1, (len(train_loader) * args.epochs) // max(1, args.grad_accum))
        warmup_steps = int(total_steps * args.warmup_ratio)

        # DeepSpeed configuration with dynamic values
        import json
        with open(args.deepspeed_config, 'r') as f:
            ds_config = json.load(f)

        # Update auto parameters (ensure they are integers/floats, not strings)
        ds_config['train_micro_batch_size_per_gpu'] = int(args.batch_size)
        ds_config['gradient_accumulation_steps'] = int(args.grad_accum)
        ds_config['train_batch_size'] = int(args.batch_size) * int(args.grad_accum) * world_size

        # Set scheduler parameters
        ds_config['scheduler']['params']['total_num_steps'] = int(total_steps)
        ds_config['scheduler']['params']['warmup_num_steps'] = int(warmup_steps)
        ds_config['scheduler']['params']['warmup_min_lr'] = float(args.min_lr)
        ds_config['scheduler']['params']['warmup_max_lr'] = float(args.lr)

        # Set optimizer LR
        ds_config['optimizer']['params']['lr'] = float(args.lr)
        ds_config['optimizer']['params']['weight_decay'] = float(args.weight_decay)

        # Initialize DeepSpeed
        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config,
        )

    # Print freeze status
    if rank == 0:
        print("\n" + "="*80)
        print("🔍 MODEL FREEZE STATUS CHECK")
        print("="*80)

        model_module = model_engine.module if isinstance(model_engine, deepspeed.DeepSpeedEngine) else model_engine

        # Check VLM
        model_to_check = model_engine.module if isinstance(model_engine, deepspeed.DeepSpeedEngine) else model_engine
        if model_to_check.vl_model is not None:
            vlm_trainable = sum(p.numel() for p in model_to_check.vl_model.parameters() if p.requires_grad)
            vlm_total = sum(p.numel() for p in model_to_check.vl_model.parameters())
            print(f"VLM (Qwen2.5-VL):        {vlm_trainable:>12,} / {vlm_total:>12,} trainable ({vlm_trainable/vlm_total*100:.1f}%)")
        else:
            print(f"VLM (Qwen2.5-VL):        Not loaded (cache_only_mode)")

        # Check Action Expert
        action_trainable = sum(p.numel() for p in model_to_check.action_expert.parameters() if p.requires_grad)
        action_total = sum(p.numel() for p in model_to_check.action_expert.parameters())
        print(f"Action Expert:           {action_trainable:>12,} / {action_total:>12,} trainable ({action_trainable/action_total*100:.1f}%)")

        # Check Sensor Encoder
        if hasattr(model_to_check, 'sensor_encoder'):
            sensor_trainable = sum(p.numel() for p in model_to_check.sensor_encoder.parameters() if p.requires_grad)
            sensor_total = sum(p.numel() for p in model_to_check.sensor_encoder.parameters())
            print(f"Sensor Encoder:          {sensor_trainable:>12,} / {sensor_total:>12,} trainable ({sensor_trainable/sensor_total*100:.1f}%)")

        # Check Robot State Encoder
        if hasattr(model_to_check, 'robot_state_encoder'):
            robot_trainable = sum(p.numel() for p in model_to_check.robot_state_encoder.parameters() if p.requires_grad)
            robot_total = sum(p.numel() for p in model_to_check.robot_state_encoder.parameters())
            print(f"Robot State Encoder:     {robot_trainable:>12,} / {robot_total:>12,} trainable ({robot_trainable/robot_total*100:.1f}%)")

        # Total
        total_trainable = sum(p.numel() for p in model_to_check.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model_to_check.parameters())
        print(f"{'-'*80}")
        print(f"TOTAL:                   {total_trainable:>12,} / {total_params:>12,} trainable ({total_trainable/total_params*100:.1f}%)")
        print("="*80 + "\n")

    # Resume from checkpoint (.pt file format)
    start_epoch = 0
    if args.resume:
        resume_path = Path(args.resume)

        # Add .pt extension if not provided
        if not resume_path.suffix:
            resume_path = resume_path.with_suffix('.pt')

        if resume_path.is_file():
            if rank == 0:
                print(f"📦 Loading checkpoint: {resume_path}")

            ckpt = torch.load(str(resume_path), map_location='cpu')
            state_dict = ckpt.get("model_state_dict", ckpt)

            # Filter out encoder keys if encoder checkpoints are provided
            # This allows loading Action Decoder from resume while keeping encoders from their own checkpoints
            exclude_prefixes = []
            if args.load_sensor_encoder_checkpoint and os.path.exists(args.load_sensor_encoder_checkpoint):
                exclude_prefixes.append('sensor_encoder.')
                if rank == 0:
                    print(f"   🔒 Excluding 'sensor_encoder.*' from resume (using separate checkpoint)")
            if args.load_robot_state_encoder_checkpoint and os.path.exists(args.load_robot_state_encoder_checkpoint):
                exclude_prefixes.append('robot_state_encoder.')
                if rank == 0:
                    print(f"   🔒 Excluding 'robot_state_encoder.*' from resume (using separate checkpoint)")

            if exclude_prefixes:
                filtered_state_dict = {}
                excluded_count = 0
                for k, v in state_dict.items():
                    if any(k.startswith(prefix) for prefix in exclude_prefixes):
                        excluded_count += 1
                        continue
                    filtered_state_dict[k] = v
                state_dict = filtered_state_dict
                if rank == 0:
                    print(f"   ℹ️ Excluded {excluded_count} encoder parameters from resume checkpoint")

            # Handle positional encoding size mismatch (65 -> 100)
            if 'robot_state_encoder.pos_encoder' in state_dict:
                pretrained_pos_enc = state_dict['robot_state_encoder.pos_encoder']
                current_pos_enc = model_engine.module.robot_state_encoder.pos_encoder

                if pretrained_pos_enc.shape[1] != current_pos_enc.shape[1]:
                    if rank == 0:
                        print(f"   🔧 Interpolating positional encoding: {pretrained_pos_enc.shape} -> {current_pos_enc.shape}")
                    pretrained_pos_enc = pretrained_pos_enc.permute(0, 2, 1)
                    interpolated = torch.nn.functional.interpolate(
                        pretrained_pos_enc, size=current_pos_enc.shape[1],
                        mode='linear', align_corners=True
                    )
                    state_dict['robot_state_encoder.pos_encoder'] = interpolated.permute(0, 2, 1)

            # Load model weights
            missing, unexpected = model_engine.module.load_state_dict(state_dict, strict=False)
            if rank == 0:
                if missing:
                    print(f"   ⚠️ Missing keys: {len(missing)}")
                if unexpected:
                    print(f"   ⚠️ Unexpected keys: {len(unexpected)}")
                print(f"   ✅ Model weights loaded (Action Decoder + VLM)")

                # Try to resume epoch if available
                if "epoch" in ckpt:
                    start_epoch = ckpt["epoch"]
                    print(f"   ℹ️ Resuming from epoch {start_epoch}")
                else:
                    print(f"   ℹ️ Starting from epoch 0")

        else:
            if rank == 0:
                print(f"   ⚠️ Checkpoint not found: {resume_path}")
                print(f"   Starting training from scratch")

    # Train with DeepSpeed
    Train_FlowMatching(
        model_engine=model_engine,
        optimizer=optimizer,
        scheduler=scheduler,
        data_loader=train_loader,
        num_epochs=args.epochs,
        grad_accum_steps=args.grad_accum,
        device=device,
        val_loader=val_loader,
        start_epoch=start_epoch,
        sensor_enabled=args.sensor_enabled,
        sensor_loss_weight=args.sensor_loss_weight,
        finetune_vl=args.finetune_vl,
    )

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
