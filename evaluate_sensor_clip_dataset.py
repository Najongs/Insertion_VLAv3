#!/usr/bin/env python3
"""

CUDA_VISIBLE_DEVICES=3 python evaluate_sensor_clip_dataset.py \
    --dataset-paths /home/najo/NAS/VLA/dataset/New_dataset4/Eye_trocar \
    --checkpoint checkpoints/sensor_clip_best.pth \
    --device cpu \
    --num-workers 0 \
    --front-progress 0.8 --back-progress 0.95 --progress-window 0.05 \
    --early-threshold 0.8 --late-threshold 0.9 \
    --progress-profile-bins 40 \
    --vlm-comparison-samples 0
      
Evaluate a SensorImage CLIP checkpoint on one or more datasets.

The script loads cached CLIP VLM features, runs the trained sensor encoder
against the same view (View5) embeddings, and reports retrieval-style metrics
plus simple visualizations.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
import glob
from PIL import Image
from TRAIN_SensorImage_CLIP import (
    CLIPModel,
    SensorImageCLIPDataset,
    clip_collate_fn,
    extract_task_name_from_episode_path,
    get_clip_prompt_hash,
    get_formatted_clip_prompt,
    infer_cached_feature_spec,
)
from models.unified_model import ForceAwareSensorEncoder, force_bn_fp32_
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from vla_datasets.unified_dataset import UnifiedVLADataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SensorImage CLIP checkpoints on cached datasets."
    )
    parser.add_argument(
        "--dataset-paths",
        type=str,
        nargs="+",
        required=True,
        help="경로 혹은 glob 패턴. episode, task 폴더, dataset root 모두 허용.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/sensor_clip_best.pth",
        help="불러올 SensorImage CLIP 체크포인트 경로.",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default="/home/najo/NAS/VLA/dataset/cache",
        help="clip_vlm_features 가 포함된 캐시 루트 경로.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="지표/시각화를 저장할 경로.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sensor-window-size", type=int, default=65)
    parser.add_argument("--sensor-output-dim", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument(
        "--intermediate-vlm-dim",
        type=int,
        default=0,
        help="VLM 중간 차원. 0이면 체크포인트에서 자동 추론.",
    )
    parser.add_argument("--sensor-dist-hidden-dim", type=int, default=128)
    parser.add_argument("--sensor-force-hidden-dim", type=int, default=48)
    parser.add_argument("--sensor-transformer-dim", type=int, default=256)
    parser.add_argument("--sensor-transformer-layers", type=int, default=1)
    parser.add_argument("--vlm-reuse-count", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--robot-window-size", type=int, default=100)
    parser.add_argument("--tail-bias", type=float, default=1.0)
    parser.add_argument(
        "--max-visualization-samples",
        type=int,
        default=400,
        help="PCA 시각화에 사용할 샘플 수.",
    )
    parser.add_argument(
        "--max-neg-samples",
        type=int,
        default=20000,
        help="음성 유사도 히스토그램 샘플 수 한도.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="연산에 사용할 디바이스.",
    )
    parser.add_argument(
        "--front-progress",
        type=float,
        default=0.80,
        help="앞쪽 비교 기준 누적 비율 (0~1).",
    )
    parser.add_argument(
        "--back-progress",
        type=float,
        default=0.95,
        help="뒤쪽 비교 기준 누적 비율 (0~1).",
    )
    parser.add_argument(
        "--progress-window",
        type=float,
        default=0.05,
        help="각 구간의 폭 (예: 0.05 → 5%p 범위 평균).",
    )
    parser.add_argument(
        "--progress-profile-bins",
        type=int,
        default=20,
        help="진행도-유사도 프로파일을 만들 때 사용할 구간 개수.",
    )
    parser.add_argument(
        "--early-threshold",
        type=float,
        default=0.8,
        help="센서가 아직 변하지 않았다고 보는 진행도 상한 (0~1).",
    )
    parser.add_argument(
        "--late-threshold",
        type=float,
        default=0.9,
        help="센서가 크게 변하기 시작했다고 보는 진행도 하한 (0~1).",
    )
    parser.add_argument(
        "--vlm-comparison-samples",
        type=int,
        default=0,
        help=">0이면 front/back 구간에서 해당 개수만큼 VLM 7B 설명을 생성.",
    )
    parser.add_argument(
        "--vlm-model-name",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="비교용으로 사용할 Qwen VLM 모델 이름.",
    )
    parser.add_argument(
        "--vlm-max-new-tokens",
        type=int,
        default=128,
        help="VLM 비교 시 생성할 최대 토큰 수.",
    )
    parser.add_argument(
        "--vlm-device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="VLM 추론에 사용할 디바이스.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0보다 크면 해당 샘플 수까지만 평가.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="결과 재현성을 위한 시드."
    )
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_episode_dirs(raw_paths: Sequence[str]) -> List[Path]:
    """Expand glob patterns and return episode directories that contain data."""
    dirs: List[Path] = []
    for entry in raw_paths:
        if any(ch in entry for ch in "*?[]"):
            expanded = sorted(Path(p) for p in glob.glob(entry))
        else:
            expanded = [Path(entry)]
        if not expanded:
            print(f"⚠️  경로를 찾을 수 없습니다: {entry}")
        for path in expanded:
            path = path.expanduser().resolve()
            if not path.exists():
                print(f"⚠️  경로가 존재하지 않습니다: {path}")
                continue
            if (path / "metadata.json").exists() or (path / "data.pkl").exists():
                dirs.append(path)
                continue
            # Treat as root (task or dataset). Collect immediate children.
            for sub in sorted(path.iterdir()):
                if not sub.is_dir():
                    continue
                if (sub / "metadata.json").exists() or (sub / "data.pkl").exists():
                    dirs.append(sub)
    # Deduplicate while preserving order
    seen = {}
    for d in dirs:
        seen.setdefault(str(d), d)
    return list(seen.values())


def build_unified_dataset(
    episode_dirs: Sequence[Path],
    args: argparse.Namespace,
    clip_cache_root: Path,
) -> ConcatDataset:
    datasets = []
    for episode_dir in episode_dirs:
        fmt = "new" if (episode_dir / "metadata.json").exists() else "old"
        try:
            ds = UnifiedVLADataset(
                data_dir=str(episode_dir),
                format=fmt,
                horizon=args.horizon,
                vlm_reuse_count=args.vlm_reuse_count,
                sensor_window_size=args.sensor_window_size,
                robot_window_size=args.robot_window_size,
                action_expert_hz=10,
                cache_root=str(clip_cache_root),
                use_cache=False,
                use_augmentation=False,
                cache_build_only=False,
            )
        except Exception as exc:
            print(f"⚠️  {episode_dir} 로딩 실패: {exc}")
            continue
        if len(ds) == 0:
            print(f"⚠️  {episode_dir} 에 사용 가능한 샘플이 없습니다.")
            continue
        datasets.append(ds)

    if not datasets:
        raise RuntimeError("평가할 데이터셋을 하나도 로드하지 못했습니다.")

    if len(datasets) == 1:
        return ConcatDataset(datasets)
    return ConcatDataset(datasets)


def infer_feature_dims_from_dataset(
    dataset: ConcatDataset, clip_cache_root: Path
) -> Tuple[int, int]:
    prompt_hashes = []
    for sub_dataset in dataset.datasets:
        task_name = extract_task_name_from_episode_path(sub_dataset.data_dir)
        prompt_hashes.append(get_clip_prompt_hash(task_name))

    checked = set()
    for prompt_hash in prompt_hashes:
        if prompt_hash in checked:
            continue
        checked.add(prompt_hash)
        spec = infer_cached_feature_spec(
            clip_cache_root, prompt_hash, return_none_if_missing=True
        )
        if spec is not None:
            image_dim, text_dim, _ = spec
            return image_dim, text_dim

    raise RuntimeError(
        f"캐시에서 feature spec 을 찾지 못했습니다. cache_root={clip_cache_root}"
    )


def build_clip_model(
    args: argparse.Namespace, image_dim: int, text_dim: int, device: torch.device
) -> CLIPModel:
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    if args.intermediate_vlm_dim > 0:
        intermediate_dim = args.intermediate_vlm_dim
    else:
        proj_weight = state_dict.get("image_feature_proj.weight")
        intermediate_dim = proj_weight.shape[0] if proj_weight is not None else 2048
        print(f"   💡 체크포인트에서 intermediate_vlm_dim={intermediate_dim} 추론")
    sensor_encoder = ForceAwareSensorEncoder(
        temporal_length=args.sensor_window_size,
        output_dim=args.sensor_output_dim,
        dist_hidden_dim=args.sensor_dist_hidden_dim,
        force_hidden_dim=args.sensor_force_hidden_dim,
        use_transformer=True,
        num_transformer_layers=args.sensor_transformer_layers,
        transformer_dim=args.sensor_transformer_dim,
    )
    force_bn_fp32_(sensor_encoder)
    model = CLIPModel(
        sensor_encoder=sensor_encoder,
        sensor_output_dim=args.sensor_output_dim,
        image_embedding_dim=image_dim,
        text_embedding_dim=text_dim,
        projection_dim=args.embedding_dim,
        intermediate_vlm_dim=intermediate_dim,
        tail_bias=args.tail_bias,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"⚠️  누락된 키: {missing}")
    if unexpected:
        print(f"⚠️  예기치 않은 키: {unexpected}")
    model.to(device)
    model.eval()
    return model


def linear_pca_2d(array: np.ndarray) -> np.ndarray:
    array = array - array.mean(axis=0, keepdims=True)
    # Use SVD for deterministic PCA
    u, s, vh = np.linalg.svd(array, full_matrices=False)
    components = vh[:2]
    return array @ components.T


def evaluate_model(
    model: CLIPModel,
    dataloader: DataLoader,
    device: torch.device,
    max_samples: int,
    max_vis: int,
    max_neg_samples: int,
    progress_cfg: dict,
):
    sensor_embeddings = []
    vlm_embeddings = []
    pos_scores: List[float] = []
    neg_scores: List[float] = []
    processed = 0
    sensor_top1 = 0
    vlm_top1 = 0
    pos_sum = 0.0
    neg_sum = 0.0
    neg_count = 0
    scale_values = []
    episode_records: dict[str, List[dict]] = {}
    sensor_vis_meta = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
            sensor = batch["sensor_data"].to(device)
            vlm_img = batch["vlm_image_features"].to(device)
            vlm_txt = batch["vlm_guidance_vector"].to(device)

            sensor_emb, vlm_emb, _, _, scale = model(sensor, vlm_img, vlm_txt)
            logits = (sensor_emb @ vlm_emb.T) * scale
            diag = logits.diag()

            batch_size = sensor_emb.size(0)
            processed += batch_size
            scale_values.append(scale.item())

            sensor_top1 += (logits.argmax(dim=1) == torch.arange(batch_size, device=logits.device)).sum().item()
            vlm_top1 += (logits.argmax(dim=0) == torch.arange(batch_size, device=logits.device)).sum().item()

            pos_sum += diag.sum().item()
            pos_scores.extend(diag.cpu().tolist())

            if batch_size > 1:
                neg_total = logits.sum(dim=1) - diag
                row_neg_mean = neg_total / (batch_size - 1)
            else:
                row_neg_mean = torch.zeros_like(diag)
            neg_sum += row_neg_mean.sum().item()
            neg_count += batch_size

            neg_matrix = logits.cpu().numpy()
            np.fill_diagonal(neg_matrix, np.nan)
            neg_flat = neg_matrix[~np.isnan(neg_matrix)]
            if neg_flat.size > 0 and len(neg_scores) < max_neg_samples:
                take = min(max_neg_samples - len(neg_scores), neg_flat.size)
                if take < neg_flat.size:
                    idx = np.random.choice(neg_flat.size, take, replace=False)
                    neg_scores.extend(neg_flat[idx].tolist())
                else:
                    neg_scores.extend(neg_flat.tolist())

            vlm_keys = batch.get("vlm_cache_keys", [None] * batch_size)
            timestamps = batch["timestamps"]
            episode_ids = batch["episode_ids"]

            if len(sensor_embeddings) < max_vis:
                remaining = max_vis - len(sensor_embeddings)
                take = min(remaining, batch_size)
                for j in range(take):
                    sensor_embeddings.append(sensor_emb[j].cpu().numpy())
                    vlm_embeddings.append(vlm_emb[j].cpu().numpy())
                    meta_key = vlm_keys[j]
                    vlm_idx = None
                    if isinstance(meta_key, (list, tuple)) and len(meta_key) >= 2:
                        try:
                            vlm_idx = int(meta_key[1])
                        except (TypeError, ValueError):
                            vlm_idx = None
                    sensor_vis_meta.append(
                        {
                            "episode_id": episode_ids[j],
                            "timestamp": float(timestamps[j]),
                            "vlm_idx": vlm_idx,
                        }
                    )

            # 기록 (episode, timestamp)
            diag_list = diag.cpu().tolist()
            for i in range(batch_size):
                meta_key = vlm_keys[i]
                vlm_idx = None
                if isinstance(meta_key, (list, tuple)) and len(meta_key) >= 2:
                    try:
                        vlm_idx = int(meta_key[1])
                    except (TypeError, ValueError):
                        vlm_idx = None
                episode_records.setdefault(episode_ids[i], []).append(
                    {
                        "timestamp": float(timestamps[i]),
                        "score": float(diag_list[i]),
                        "vlm_idx": vlm_idx,
                    }
                )

            if max_samples and processed >= max_samples:
                break

    if processed == 0:
        raise RuntimeError("데이터를 처리하지 못했습니다.")

    metrics = {
        "samples": processed,
        "sensor_top1": sensor_top1 / processed,
        "vlm_top1": vlm_top1 / processed,
        "pos_mean": pos_sum / processed,
        "neg_mean": neg_sum / max(neg_count, 1),
        "scale_mean": float(np.mean(scale_values)) if scale_values else 1.0,
        "scale_std": float(np.std(scale_values)) if scale_values else 0.0,
    }

    sensor_arr = np.stack(sensor_embeddings) if sensor_embeddings else None
    vlm_arr = np.stack(vlm_embeddings) if vlm_embeddings else None
    segment_info, progress_samples, progress_lookup = analyze_progress_segments(
        episode_records, progress_cfg
    )
    metrics["progress_segments"] = segment_info
    vis_payload = {
        "sensor": sensor_arr,
        "vlm": vlm_arr,
        "pos_scores": pos_scores,
        "neg_scores": neg_scores,
        "progress_points": progress_samples,
        "progress_ranges": {
            "front": [progress_cfg["front_start"], progress_cfg["front_end"]],
            "back": [progress_cfg["back_start"], progress_cfg["back_end"]],
        },
        "per_episode_segments": segment_info.get("per_episode", {}),
        "sensor_meta": sensor_vis_meta,
        "progress_lookup": progress_lookup,
    }
    return metrics, vis_payload, progress_samples


def analyze_progress_segments(
    episode_records: dict,
    cfg: dict,
) -> Tuple[dict, List[dict], dict]:
    front_scores = []
    back_scores = []
    per_episode = {}
    progress_samples: List[dict] = []
    progress_lookup = {}

    front_range = (cfg["front_start"], cfg["front_end"])
    back_range = (cfg["back_start"], cfg["back_end"])

    for episode, entries in episode_records.items():
        entries.sort(key=lambda x: x["timestamp"] if isinstance(x, dict) else x[0])
        n = len(entries)
        ep_front = []
        ep_back = []
        for idx, entry in enumerate(entries, start=1):
            if isinstance(entry, tuple):
                timestamp, score = entry
                vlm_idx = None
            else:
                timestamp = entry["timestamp"]
                score = entry["score"]
                vlm_idx = entry.get("vlm_idx")
            progress = idx / n if n > 0 else 0.0
            key = (episode, timestamp)
            progress_lookup[key] = progress
            progress_samples.append(
                {
                    "episode_id": episode,
                    "timestamp": timestamp,
                    "progress": progress,
                    "score": score,
                    "vlm_idx": vlm_idx,
                }
            )
            if front_range[0] <= progress < front_range[1]:
                ep_front.append(score)
            if back_range[0] <= progress < back_range[1]:
                ep_back.append(score)
        per_episode[episode] = {
            "front_count": len(ep_front),
            "front_mean": float(np.mean(ep_front)) if ep_front else None,
            "back_count": len(ep_back),
            "back_mean": float(np.mean(ep_back)) if ep_back else None,
        }
        front_scores.extend(ep_front)
        back_scores.extend(ep_back)

    summary = {
        "front": {
            "range": list(front_range),
            "count": len(front_scores),
            "mean": float(np.mean(front_scores)) if front_scores else None,
        },
        "back": {
            "range": list(back_range),
            "count": len(back_scores),
            "mean": float(np.mean(back_scores)) if back_scores else None,
        },
        "per_episode": per_episode,
    }
    return summary, progress_samples, progress_lookup


def compute_progress_profile(
    progress_samples: List[dict], bin_count: int
) -> List[dict]:
    if not progress_samples or bin_count <= 0:
        return []

    bin_edges = np.linspace(0.0, 1.0, bin_count + 1)
    profile = []
    scores = np.array([item["score"] for item in progress_samples])
    progresses = np.array([item["progress"] for item in progress_samples])

    for idx in range(bin_count):
        start = bin_edges[idx]
        end = bin_edges[idx + 1]
        mask = (progresses >= start) & (
            progresses < end if idx < bin_count - 1 else progresses <= end + 1e-8
        )
        if not np.any(mask):
            profile.append(
                {
                    "start": float(start),
                    "end": float(end),
                    "count": 0,
                    "mean": None,
                }
            )
            continue
        mean_val = float(scores[mask].mean())
        profile.append(
            {
                "start": float(start),
                "end": float(end),
                "count": int(mask.sum()),
                "mean": mean_val,
            }
        )
    return profile


def summarize_early_late_progress(
    progress_samples: List[dict], early_thresh: float, late_thresh: float
) -> dict:
    if not progress_samples:
        return {}
    early_thresh = np.clip(early_thresh, 0.0, 1.0)
    late_thresh = np.clip(late_thresh, 0.0, 1.0)
    if late_thresh < early_thresh:
        late_thresh = early_thresh

    early_scores = []
    middle_scores = []
    late_scores = []
    for item in progress_samples:
        prog = item["progress"]
        score = item["score"]
        if prog < early_thresh:
            early_scores.append(score)
        elif prog >= late_thresh:
            late_scores.append(score)
        else:
            middle_scores.append(score)

    def _summary(values):
        return {
            "count": len(values),
            "mean": float(np.mean(values)) if values else None,
        }

    return {
        "early_range": [0.0, float(early_thresh)],
        "late_range": [float(late_thresh), 1.0],
        "early": _summary(early_scores),
        "middle": _summary(middle_scores),
        "late": _summary(late_scores),
    }


def plot_visualizations(
    vis_data: dict,
    save_path: Path,
    dataset_label: str,
):
    if vis_data["sensor"] is None or vis_data["vlm"] is None:
        print("⚠️  시각화를 위한 임베딩이 부족하여 플롯을 건너뜁니다.")
        return

    sensor_2d = linear_pca_2d(vis_data["sensor"])
    vlm_2d = linear_pca_2d(vis_data["vlm"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax0, ax1, ax2 = axes

    sensor_meta = vis_data.get("sensor_meta", [])
    progress_lookup = vis_data.get("progress_lookup", {})
    colors = None
    if sensor_meta and len(sensor_meta) == len(sensor_2d):
        colors = [
            progress_lookup.get((meta["episode_id"], meta["timestamp"]), 0.0)
            for meta in sensor_meta
        ]
    sensor_scatter = ax0.scatter(
        sensor_2d[:, 0],
        sensor_2d[:, 1],
        s=18,
        alpha=0.8,
        c=colors if colors is not None else "#4c72b0",
        cmap="viridis",
        label="Sensor",
    )
    ax0.scatter(vlm_2d[:, 0], vlm_2d[:, 1], s=18, alpha=0.7, label="VLM")
    max_lines = min(40, len(sensor_2d))
    for i in range(max_lines):
        ax0.plot(
            [sensor_2d[i, 0], vlm_2d[i, 0]],
            [sensor_2d[i, 1], vlm_2d[i, 1]],
            color="gray",
            alpha=0.2,
            linewidth=0.8,
        )
    ax0.set_title(f"PCA Embeddings ({dataset_label})")
    ax0.set_xlabel("PC1")
    ax0.set_ylabel("PC2")
    if colors is not None:
        cbar = fig.colorbar(sensor_scatter, ax=ax0, fraction=0.046, pad=0.04)
        cbar.set_label("Progress (0=start,1=end)")
    ax0.legend(loc="best")
    ax0.grid(alpha=0.2)

    pos = vis_data["pos_scores"]
    neg = vis_data["neg_scores"]
    if pos:
        ax1.hist(pos, bins=40, alpha=0.7, label="Positive", color="#4c72b0")
    if neg:
        ax1.hist(neg, bins=40, alpha=0.7, label="Negative", color="#dd8452")
    ax1.set_title("Similarity Distribution (scaled logits)")
    ax1.set_xlabel("Score")
    ax1.set_ylabel("Count")
    ax1.legend(loc="best")
    ax1.grid(alpha=0.2)

    progress_points = vis_data.get("progress_points", [])
    ranges = vis_data.get("progress_ranges", {})
    profile = vis_data.get("progress_profile", [])
    early_thresh = vis_data.get("early_threshold")
    late_thresh = vis_data.get("late_threshold")
    if progress_points:
        arr = np.array([[item["progress"], item["score"]] for item in progress_points])
        arr = arr[arr[:, 0].argsort()]
        ax2.plot(arr[:, 0], arr[:, 1], color="#4c72b0", alpha=0.3, linewidth=1.0)
        ax2.scatter(arr[:, 0], arr[:, 1], s=14, alpha=0.6, label="Samples")
        front = ranges.get("front")
        back = ranges.get("back")
        if front:
            ax2.axvspan(front[0], front[1], color="#4c72b0", alpha=0.15, label="Front Window")
        if back:
            ax2.axvspan(back[0], back[1], color="#dd8452", alpha=0.15, label="Back Window")
        if early_thresh is not None and 0 < early_thresh < 1:
            ax2.axvspan(
                0.0,
                early_thresh,
                color="#bab0ab",
                alpha=0.1,
                label=f"≤{early_thresh*100:.0f}% zone",
            )
        if late_thresh is not None and 0 < late_thresh <= 1:
            ax2.axvspan(
                late_thresh,
                1.0,
                color="#ff9da6",
                alpha=0.12,
                label=f"≥{late_thresh*100:.0f}% zone",
            )
        if profile:
            centers = []
            means = []
            for bin_info in profile:
                if bin_info.get("mean") is None:
                    continue
                centers.append(
                    (bin_info["start"] + bin_info["end"]) / 2.0
                )
                means.append(bin_info["mean"])
            if centers:
                ax2.plot(
                    centers,
                    means,
                    color="#c44e52",
                    linewidth=2.2,
                    label="Bin means",
                )
        ax2.set_xlim(0, 1.01)
        ax2.set_xlabel("Normalized Progress")
        ax2.set_ylabel("Similarity")
        ax2.set_title("Progress vs Similarity")
        ax2.grid(alpha=0.2)
        handles, labels = ax2.get_legend_handles_labels()
        if handles:
            uniq = {}
            for h, lbl in zip(handles, labels):
                if lbl not in uniq:
                    uniq[lbl] = h
            ax2.legend(list(uniq.values()), list(uniq.keys()), loc="best")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def select_progress_samples(
    progress_samples: List[dict],
    front_range: Tuple[float, float],
    back_range: Tuple[float, float],
    total_count: int,
) -> List[dict]:
    if total_count <= 0 or not progress_samples:
        return []
    front = [
        sample
        for sample in progress_samples
        if front_range[0] <= sample["progress"] < front_range[1]
    ]
    back = [
        sample
        for sample in progress_samples
        if back_range[0] <= sample["progress"] < back_range[1]
    ]
    per_side = max(1, total_count // 2)
    selected = front[:per_side] + back[:per_side]
    if len(selected) < total_count:
        remaining = [
            sample for sample in progress_samples if sample not in selected
        ]
        selected.extend(remaining[: total_count - len(selected)])
    return selected[:total_count]


def _extract_timestamp_from_name(name: str) -> float:
    try:
        return float(name)
    except ValueError:
        parts = name.split("_")
        for part in reversed(parts):
            try:
                return float(part)
            except ValueError:
                continue
    return 0.0


def load_view5_image(episode_dir: Path, vlm_idx: int) -> Image.Image:
    candidates = [
        episode_dir / "View5",
        episode_dir / "images" / "View5",
    ]
    files = []
    for view_dir in candidates:
        if view_dir.exists():
            imgs = sorted(
                list(view_dir.glob("*.jpg")) + list(view_dir.glob("*.png")),
                key=lambda p: _extract_timestamp_from_name(p.stem),
            )
            if imgs:
                files = imgs
                break
    if not files:
        raise FileNotFoundError(f"View5 이미지를 찾을 수 없습니다: {episode_dir}")
    idx = min(max(vlm_idx, 0), len(files) - 1)
    return Image.open(files[idx]).convert("RGB")


def generate_vlm_response(image, prompt, vlm_model, vlm_processor, max_new_tokens):
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
        }
    ]
    text_input = vlm_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    model_inputs = vlm_processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device=vlm_model.device, dtype=vlm_model.dtype)
    input_lens = [len(ids) for ids in model_inputs.input_ids]
    with torch.no_grad():
        generated_ids = vlm_model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
    trimmed = [ids[len_:] for ids, len_ in zip(generated_ids, input_lens)]
    response = vlm_processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant", 1)[-1]
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>", 1)[0]
    return response.strip()


def run_vlm_comparison(
    samples: List[dict],
    args: argparse.Namespace,
    episode_dir_map: dict,
    task_name_map: dict,
) -> List[dict]:
    if not samples:
        return []

    device_type = args.vlm_device
    if device_type == "auto":
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    vlm_device = torch.device(device_type)

    print(f"⏳ VLM 비교용 모델 로딩 중... ({args.vlm_model_name}, device={vlm_device})")
    torch_dtype = torch.bfloat16 if vlm_device.type == "cuda" else torch.float32
    vlm_processor = AutoProcessor.from_pretrained(args.vlm_model_name, trust_remote_code=True)
    vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.vlm_model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=None,
    ).to(vlm_device)
    vlm_model.eval()

    results = []
    for sample in samples:
        episode_id = sample["episode_id"]
        vlm_idx = sample.get("vlm_idx")
        if vlm_idx is None:
            continue
        episode_dir = episode_dir_map.get(episode_id)
        if episode_dir is None:
            continue
        try:
            image = load_view5_image(episode_dir, vlm_idx)
        except Exception as exc:
            print(f"⚠️  {episode_id} 이미지 로드 실패: {exc}")
            continue
        task_name = task_name_map.get(episode_id, "Unknown")
        prompt = get_formatted_clip_prompt(task_name)
        try:
            response = generate_vlm_response(
                image, prompt, vlm_model, vlm_processor, args.vlm_max_new_tokens
            )
        except Exception as exc:
            print(f"⚠️  VLM 추론 실패 ({episode_id}): {exc}")
            response = f"[ERROR] {exc}"
        result_entry = dict(sample)
        result_entry["vlm_response"] = response
        results.append(result_entry)

    return results


def main():
    args = parse_args()
    set_random_seed(args.seed)

    device_type = args.device
    if device_type == "auto":
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)

    clip_cache_root = Path(args.cache_root) / "clip_vlm_features"
    if not clip_cache_root.exists():
        raise RuntimeError(f"clip_vlm_features 가 {clip_cache_root} 에 없습니다.")

    episode_dirs = resolve_episode_dirs(args.dataset_paths)
    if not episode_dirs:
        raise RuntimeError("입력 경로에서 episode 를 찾지 못했습니다.")
    print(f"📁 평가 대상 episode 수: {len(episode_dirs)}")

    unified_dataset = build_unified_dataset(episode_dirs, args, clip_cache_root)
    episode_dir_map = {}
    task_name_map = {}
    for sub_dataset in unified_dataset.datasets:
        episode_id = sub_dataset.data_dir.name
        episode_dir_map[episode_id] = sub_dataset.data_dir
        task_name_map[episode_id] = extract_task_name_from_episode_path(sub_dataset.data_dir)
    clip_dataset = SensorImageCLIPDataset(
        unified_dataset,
        vlm_annotations={},
        cache_path=None,
        clip_cache_root=str(clip_cache_root),
        mode="train",
        force_on_the_fly=False,
        skip_cache_verification=False,
    )
    clip_dataset.eval()

    feature_dims = infer_feature_dims_from_dataset(unified_dataset, clip_cache_root)
    print(f"✅ Cached feature spec - image_dim={feature_dims[0]}, text_dim={feature_dims[1]}")

    model = build_clip_model(args, feature_dims[0], feature_dims[1], device)

    collate = lambda batch: clip_collate_fn(
        batch,
        window_size=args.sensor_window_size,
        vlm_model=None,
        vlm_processor=None,
        clip_cache_manager=None,
        device=device,
    )
    dataloader = DataLoader(
        clip_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    front_end = min(1.0, args.front_progress + args.progress_window)
    back_end = min(1.0, args.back_progress + args.progress_window)
    progress_cfg = {
        "front_start": max(0.0, args.front_progress),
        "front_end": max(front_end, args.front_progress),
        "back_start": max(0.0, args.back_progress),
        "back_end": max(back_end, args.back_progress),
    }

    metrics, vis_payload, progress_samples = evaluate_model(
        model,
        dataloader,
        device,
        max_samples=args.max_samples,
        max_vis=args.max_visualization_samples,
        max_neg_samples=args.max_neg_samples,
        progress_cfg=progress_cfg,
    )
    progress_profile = compute_progress_profile(
        progress_samples, args.progress_profile_bins
    )
    early_late = summarize_early_late_progress(
        progress_samples, args.early_threshold, args.late_threshold
    )
    metrics["progress_profile"] = progress_profile
    metrics["early_late_segments"] = early_late
    vis_payload["progress_profile"] = progress_profile
    vis_payload["early_threshold"] = args.early_threshold
    vis_payload["late_threshold"] = args.late_threshold

    vlm_results = []
    if args.vlm_comparison_samples > 0:
        front_range = (progress_cfg["front_start"], progress_cfg["front_end"])
        back_range = (progress_cfg["back_start"], progress_cfg["back_end"])
        selected_samples = select_progress_samples(
            progress_samples, front_range, back_range, args.vlm_comparison_samples
        )
        vlm_results = run_vlm_comparison(
            selected_samples, args, episode_dir_map, task_name_map
        )
        metrics["vlm_comparison"] = vlm_results

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parent_names = {d.parent.name for d in episode_dirs}
    dataset_label = parent_names.pop() if len(parent_names) == 1 else "multi"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"sensor_clip_eval_{dataset_label}_{timestamp}"
    fig_path = output_dir / f"{base_name}.png"
    json_path = output_dir / f"{base_name}.json"

    plot_visualizations(vis_payload, fig_path, dataset_label)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n=== Evaluation Summary ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:>12}: {value:.4f}")
        else:
            print(f"{key:>12}: {value}")
    seg = metrics.get("progress_segments")
    if seg:
        front = seg.get("front", {})
        back = seg.get("back", {})
        if front.get("mean") is not None:
            print(
                f" Front {front['range'][0]*100:.1f}-{front['range'][1]*100:.1f}%: "
                f"mean={front['mean']:.4f} (n={front['count']})"
            )
        if back.get("mean") is not None:
            print(
                f"  Back {back['range'][0]*100:.1f}-{back['range'][1]*100:.1f}%: "
                f"mean={back['mean']:.4f} (n={back['count']})"
            )
    early_late = metrics.get("early_late_segments") or {}
    early = early_late.get("early")
    late = early_late.get("late")
    if early:
        erange = early_late.get("early_range", [0, 0])
        if early.get("mean") is not None:
            print(
                f" Early ≤{erange[1]*100:.0f}%: mean={early['mean']:.4f} (n={early['count']})"
            )
    if late:
        lrange = early_late.get("late_range", [0, 1])
        if late.get("mean") is not None:
            print(
                f"  Late ≥{lrange[0]*100:.0f}%: mean={late['mean']:.4f} (n={late['count']})"
            )
    if vlm_results:
        print("\n🗒️  VLM 비교 샘플:")
        for item in vlm_results:
            print(
                f" - {item['episode_id']} | progress {item['progress']*100:.1f}% "
                f"| score {item['score']:.3f} | idx {item.get('vlm_idx')}: {item['vlm_response'][:160]}"
            )
    print(f"\n📈 시각화 저장: {fig_path}")
    print(f"📝 지표 저장: {json_path}")


if __name__ == "__main__":
    main()
