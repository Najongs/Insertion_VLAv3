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
            first_task_dir = next(d for d in new_path.iterdir() if d.is_dir())
            first_episode_dir = next(d for d in first_task_dir.iterdir() if d.is_dir() and (d.name.startswith('episode_') or d.name.startswith('data_collection_')))

            ds = UnifiedVLADataset(
                data_dir=str(first_episode_dir),
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
    model,
    data_loader,
    optimizer,
    num_epochs=3,
    grad_accum_steps=8,
    device="cuda",
    scheduler=None,
    sched_on="step",
    val_loader=None,
    start_epoch=0,
    sensor_enabled=True,
    sensor_loss_weight=2.0,
):
    """Flow Matching training loop - OPTIMIZED"""
    rank = dist.get_rank()
    writer = AsyncCheckpointWriter(max_queue=2, sync_every=0) if rank == 0 else None

    model.train()
    if rank == 0:
        wandb.init(
            project="QwenVLA-FlowMatching",
            name=f"flow_matching_{time.strftime('%m%d_%H%M')}",
            resume="allow",
            id=f"qvla_flow_matching_{int(time.time())}",
            settings=wandb.Settings(start_method="thread", _disable_stats=True),
            config={
                "model_type": "flow_matching",
                "lr": optimizer.param_groups[0]["lr"],
                "grad_accum_steps": grad_accum_steps,
                "epochs": num_epochs,
                "scheduler": sched_on,
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

        optimizer.zero_grad(set_to_none=True)
        model.train()

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

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # Flow matching: model returns loss directly
                    flow_loss, _, _ = model(
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

                    loss = flow_loss / grad_accum_steps

                sync_context = model.no_sync() if (step + 1) % grad_accum_steps != 0 else nullcontext()
                with sync_context:
                    loss.backward()

                total_loss += loss.item() * grad_accum_steps

                if (step + 1) % grad_accum_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    if scheduler is not None and sched_on == "step":
                        scheduler.step()

                    global_step += 1

                    lr = optimizer.param_groups[0]["lr"]
                    if rank == 0:
                        running_cache_ratio = (
                            epoch_cache_hits / epoch_cache_total
                            if epoch_cache_total > 0 else 0.0
                        )
                        cache_status = (
                            f"{running_cache_ratio*100:.1f}% ({epoch_cache_hits}/{epoch_cache_total})"
                            if epoch_cache_total > 0 else "0.0% (0/0)"
                        )
                        postfix_dict = {
                            "loss": f"{loss.item() * grad_accum_steps:.6f}",
                            "lr": f"{lr:.2e}",
                            "grad": f"{grad_norm:.2f}",
                            "cache": cache_status,
                        }
                        if sensor_enabled:
                            postfix_dict["sensor"] = f"{total_sensor_samples}/{total_sensor_samples+total_nonsensor_samples}"
                        pbar.set_postfix(postfix_dict)

                        log_dict = {
                            "train/loss_step": loss.item() * grad_accum_steps,
                            "train/lr": lr,
                            "train/grad_norm": grad_norm,
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

                if (step + 1) % grad_accum_steps == 0:
                    optimizer.zero_grad(set_to_none=True)
                continue

        avg_loss_tensor = torch.tensor(total_loss / len(data_loader), device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = avg_loss_tensor.item()

        if scheduler is not None and sched_on == "epoch":
            scheduler.step()

        # Validation
        val_loss = None
        if val_loader is not None:
            model.eval()
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
                        loss, _, _ = model(
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
            model.train()

        synced_hits, synced_total = _sync_cache_stats(epoch_cache_hits, epoch_cache_total, device)
        cache_hit_ratio = (synced_hits / synced_total) if synced_total > 0 else 0.0

        # Checkpoint saving
        if rank == 0:
            import psutil, gc
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            frozen = total_params - trainable

            gpu_mem = torch.cuda.memory_allocated()/1e9
            cpu_mem = psutil.virtual_memory().percent
            gc.collect()

            log_dict = {
                "epoch": epoch + 1,
                "train/loss_epoch": avg_loss,
                "val/loss_epoch": val_loss if val_loss else None,
                "params/trainable_M": trainable / 1e6,
                "params/frozen_M": frozen / 1e6,
                "params/frozen_ratio": frozen / total_params,
                "system/gpu_mem_GB": gpu_mem,
                "system/cpu_mem_%": cpu_mem,
                "lr/base_lr": optimizer.param_groups[0]["lr"],
                "train/cache_hit_ratio": cache_hit_ratio,
                "train/cache_samples": synced_total,
            }

            if sensor_enabled:
                log_dict["train/epoch_sensor_samples"] = total_sensor_samples
                log_dict["train/epoch_nonsensor_samples"] = total_nonsensor_samples
                log_dict["train/sensor_ratio"] = total_sensor_samples / max(1, total_sensor_samples + total_nonsensor_samples)

            wandb.log(log_dict)
            print(f"\n📊 Epoch {epoch+1} Summary | Train: {avg_loss:.8f} | " +
                  (f"Val: {val_loss:.8f}" if val_loss else ""))
            print(f"   Cache hit ratio: {cache_hit_ratio*100:.2f}% ({int(synced_hits)}/{int(synced_total)})")

            model_module = model.module if hasattr(model, "module") else model
            ckpt_data = {
                "epoch": epoch + 1,
                "model_state_dict": model_module.state_dict(),
                "sensor_encoder": model_module.sensor_encoder.state_dict() if sensor_enabled else None,
                "action_expert": model_module.action_expert.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "val_loss": val_loss,
            }

            is_best = val_loss is not None and val_loss < best_val_loss

            if is_best:
                best_val_loss = val_loss
                best_path = CKPT_DIR / "flow_matching_best.pt"
                torch.save(ckpt_data, best_path)
                print(f"🏆 [Best] Validation improved → saved to {best_path}")
            else:
                latest_path = CKPT_DIR / "flow_matching_latest.pt"
                tmp_path = latest_path.with_suffix(".tmp")
                torch.save(ckpt_data, tmp_path)
                os.replace(tmp_path, latest_path)
                print(f"💾 Latest checkpoint updated: {latest_path}")

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
    parser.add_argument('--sensor_lr', type=float, default=5e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument('--hold_ratio', type=float, default=0.02)
    parser.add_argument('--sched_on', type=str, choices=['step', 'epoch'], default='step')

    # Sensor options
    parser.add_argument('--sensor_enabled', action='store_true', default=True,
                        help='Enable sensor encoder training')
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
    parser.add_argument('--freeze_encoders', action='store_true', help='Freeze sensor and robot state encoders after loading weights.')

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
            print("--- Starting VL Cache Building ---")

        # 1. Dataloader (use_cache=False is crucial)
        train_loader, _ = build_dataloaders(
            args,
            rank,
            world_size,
            use_cache=True,  # 캐시 생성 시점과 학습 시점의 프롬프트/해시를 일치시켜 재사용 가능하게 함
            cache_build_only=cache_loader_mode,
        )

        # 2. Model (needed for its processor and VL model)
        model = QwenVLAUnified(
            model_type='flow_matching', vl_model_name=vl_model_name, action_dim=7, horizon=8,
            hidden_dim=1024, sensor_enabled=False, # Sensor data not needed for VL cache
            finetune_vl='none',
            image_resize_height=args.image_resize_height, image_resize_width=args.image_resize_width,
            device_map=None,
            external_cache_root=args.cache_root,
        )
        model = model.to(device)
        
        # Pass cache_dir to the model so the builder can find it
        # This is a bit of a hack, but it's how the cache builder is designed
        model.cache_dir = cache_dir

        # 3. Run the cache building process
        build_vl_cache_distributed_optimized(
            model=model,
            dataset=train_loader.dataset,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        if rank == 0:
            print("--- VL Cache Building Complete ---")
        
        # Ensure all processes sync up before exiting
        if dist.is_initialized():
            dist.barrier()
        
        return

    # Training mode
    if rank == 0:
        print(f"⚠️ [주의] VL 캐시 검사를 건너뜁니다.")
        print(f"   캐시 경로: {cache_dir}")
        if not cache_dir.exists() or not any(cache_dir.iterdir()):
            print(f"   [경고!] 캐시 디렉토리가 비어있거나 존재하지 않습니다!")

    train_loader, val_loader = build_dataloaders(args, rank, world_size, use_cache=args.use_cache, cache_build_only=args.cache_loader_only)

    if rank == 0: print("⏳ Initializing model for training...")

    # Cache 사용 시 VLM 로드 스킵 (메모리 절약)
    use_cache_only_mode = args.use_cache and args.finetune_vl == 'none'

    model = QwenVLAUnified(
        model_type='flow_matching', vl_model_name=vl_model_name, action_dim=7, horizon=8,
        hidden_dim=1024, sensor_enabled=args.sensor_enabled,
        sensor_encoder_type='force_aware', # Match pre-training architecture
        sensor_input_channels=1026,
        sensor_temporal_length=65, sensor_output_dim=1024, robot_state_enabled=args.sensor_enabled,
        robot_state_output_dim=512, # Match pre-training architecture
        finetune_vl=args.finetune_vl,
        image_resize_height=args.image_resize_height, image_resize_width=args.image_resize_width,
        device_map=None,
        external_cache_root=args.cache_root,
        cache_only_mode=use_cache_only_mode,  # VLM freeze + cache 사용 시 VLM 로드 스킵
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

    # DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # Freeze encoders after DDP wrapping and weight loading
    if args.freeze_encoders:
        if rank == 0: print("🧊 Freezing Sensor and Robot State Encoders...")
        for param in model.module.sensor_encoder.parameters():
            param.requires_grad = False
        for param in model.module.robot_state_encoder.parameters():
            param.requires_grad = False
        if rank == 0: print("✅ Encoders frozen.")

    # When doing LoRA fine-tuning, freeze everything except VLM (Action Expert included)
    if args.finetune_vl == 'lora':
        if rank == 0: print("🧊 LoRA mode: Freezing Action Expert (only VLM LoRA adapters will be trained)...")
        for param in model.module.action_expert.parameters():
            param.requires_grad = False
        # Also freeze sensor encoders if not already frozen
        if not args.freeze_encoders:
            for param in model.module.sensor_encoder.parameters():
                param.requires_grad = False
            for param in model.module.robot_state_encoder.parameters():
                param.requires_grad = False
        if rank == 0: print("✅ Action Expert frozen. Only VLM LoRA adapters are trainable.")

    # Optimizer (created AFTER freezing)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95),
    )

    # Scheduler
    total_steps = (len(train_loader) * args.epochs) // args.grad_accum
    scheduler = build_trapezoid_scheduler(
        optimizer, total_steps=total_steps, base_lr=args.lr, min_lr=args.min_lr,
        warmup_ratio=args.warmup_ratio, hold_ratio=args.hold_ratio,
    )

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        if rank == 0: print(f"Resuming from {args.resume}")
        ckpt = copy_to_local_then_load(Path(args.resume), map_location=device)

        # Handle positional encoding size mismatch (65 -> 100)
        state_dict = ckpt["model_state_dict"]
        has_size_mismatch = False
        if 'robot_state_encoder.pos_encoder' in state_dict:
            pretrained_pos_enc = state_dict['robot_state_encoder.pos_encoder']  # [1, 65, 256]
            current_pos_enc = model.module.robot_state_encoder.pos_encoder  # [1, 100, 256]

            if pretrained_pos_enc.shape[1] != current_pos_enc.shape[1]:
                has_size_mismatch = True
                if rank == 0:
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
                state_dict['robot_state_encoder.pos_encoder'] = interpolated.permute(0, 2, 1)  # [1, 100, 256]

        # Load model state, but be careful about frozen parts
        model.module.load_state_dict(state_dict, strict=False)

        # Check if we're switching to LoRA mode (new trainable parameters added)
        switching_to_lora = args.finetune_vl == 'lora'

        # Only load optimizer state if there's no size mismatch AND not switching to LoRA
        # (LoRA adds new trainable parameters, so optimizer state won't match)
        skip_optimizer = has_size_mismatch or switching_to_lora

        if not skip_optimizer:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if scheduler and ckpt.get("scheduler_state_dict"):
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            if rank == 0:
                if switching_to_lora:
                    print(f"   🔧 Switching to LoRA fine-tuning mode - optimizer will start fresh with new LoRA parameters.")
                else:
                    print(f"   ⚠️ Skipping optimizer/scheduler state loading due to size mismatch. Training will start fresh.")

        start_epoch = ckpt.get("epoch", 0) if not skip_optimizer else 0

    # Train
    Train_FlowMatching(
        model=model, data_loader=train_loader, optimizer=optimizer, num_epochs=args.epochs,
        grad_accum_steps=args.grad_accum, device=device, scheduler=scheduler, sched_on=args.sched_on,
        val_loader=val_loader, start_epoch=start_epoch, sensor_enabled=args.sensor_enabled,
        sensor_loss_weight=args.sensor_loss_weight,
    )

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
