#!/usr/bin/env python3
"""
Comprehensive Flow Matching VLA Model Evaluation Script

Evaluates trained VLA checkpoints on multiple episodes and provides:
- Per-episode and overall metrics (RMSE, MAE, Success Rate)
- Detailed per-dimension analysis (position, rotation, gripper)
- Trajectory visualization (GT vs Predicted)
- JSON export for further analysis
- Optional wandb logging

Usage:
    # Evaluate on specific episodes
    python EVAL_FlowMatching.py \
        --checkpoint checkpoints/flow_matching_best.pt \
        --dataset-paths "/home/najo/NAS/VLA/dataset/New_dataset6/Red_point/data_collection_*" \
        --output-dir evaluation_results/flow_best_eval \
        --batch-size 4 \
        --wandb-project "QwenVLA-Evaluation"

    # Evaluate on validation split from training data
    python EVAL_FlowMatching.py \
        --checkpoint checkpoints/flow_matching_best.pt \
        --dataset-paths "/home/najo/NAS/VLA/dataset/New_dataset6/*_point" \
        --val-split 0.1 \
        --sample-episodes 10 \
        --output-dir evaluation_results/val_eval
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import wandb

from models.unified_model import QwenVLAUnified
from vla_datasets.unified_dataset import UnifiedVLADataset, unified_collate_fn


ACTION_LABELS = ["dx", "dy", "dz", "rx", "ry", "rz", "gripper"]
POSITION_DIMS = [0, 1, 2]  # x, y, z
ROTATION_DIMS = [3, 4, 5]  # roll, pitch, yaw
GRIPPER_DIM = 6


def get_model_info(model: QwenVLAUnified) -> Dict:
    """
    Calculate model size and parameter information.

    Returns:
        Dictionary with model statistics including:
        - total_params: Total number of parameters
        - trainable_params: Number of trainable parameters
        - model_size_mb: Total model size in MB
        - component_params: Per-component parameter breakdown
    """
    def count_parameters(module):
        """Count total and trainable parameters in a module."""
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    def get_size_mb(module):
        """Calculate module size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in module.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in module.buffers())
        return (param_size + buffer_size) / (1024 ** 2)

    # Overall stats
    total_params, trainable_params = count_parameters(model)
    model_size_mb = get_size_mb(model)

    # Component breakdown
    component_params = {}

    # VL Model
    if hasattr(model, 'vl_model'):
        vl_total, vl_trainable = count_parameters(model.vl_model)
        component_params['vl_model'] = {
            'total': vl_total,
            'trainable': vl_trainable,
            'size_mb': get_size_mb(model.vl_model)
        }

    # Sensor Encoder
    if hasattr(model, 'sensor_encoder') and model.sensor_encoder is not None:
        sensor_total, sensor_trainable = count_parameters(model.sensor_encoder)
        component_params['sensor_encoder'] = {
            'total': sensor_total,
            'trainable': sensor_trainable,
            'size_mb': get_size_mb(model.sensor_encoder)
        }

    # Robot State Encoder
    if hasattr(model, 'robot_state_encoder') and model.robot_state_encoder is not None:
        robot_total, robot_trainable = count_parameters(model.robot_state_encoder)
        component_params['robot_state_encoder'] = {
            'total': robot_total,
            'trainable': robot_trainable,
            'size_mb': get_size_mb(model.robot_state_encoder)
        }

    # Flow Matching Decoder
    if hasattr(model, 'flow_decoder'):
        flow_total, flow_trainable = count_parameters(model.flow_decoder)
        component_params['flow_decoder'] = {
            'total': flow_total,
            'trainable': flow_trainable,
            'size_mb': get_size_mb(model.flow_decoder)
        }

    return {
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
        'non_trainable_params': int(total_params - trainable_params),
        'model_size_mb': float(model_size_mb),
        'model_size_gb': float(model_size_mb / 1024),
        'component_params': component_params,
    }


def print_model_info(model_info: Dict):
    """Print model information in a readable format."""
    print("\n" + "="*80)
    print("📊 MODEL INFORMATION")
    print("="*80)

    # Overall stats
    total_params = model_info['total_params']
    trainable_params = model_info['trainable_params']
    non_trainable = model_info['non_trainable_params']
    size_mb = model_info['model_size_mb']
    size_gb = model_info['model_size_gb']

    print(f"Total Parameters:        {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable Parameters:    {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"Non-trainable Parameters: {non_trainable:,} ({non_trainable/1e6:.2f}M)")

    if size_gb >= 1.0:
        print(f"Model Size:              {size_gb:.2f} GB")
    else:
        print(f"Model Size:              {size_mb:.2f} MB")

    # Component breakdown
    if model_info['component_params']:
        print("\nComponent Breakdown:")
        print("-" * 80)

        for comp_name, comp_info in model_info['component_params'].items():
            comp_label = comp_name.replace('_', ' ').title()
            comp_params = comp_info['total']
            comp_trainable = comp_info['trainable']
            comp_size = comp_info['size_mb']

            print(f"  {comp_label:25s}: {comp_params:>12,} params ({comp_params/1e6:>6.2f}M) | "
                  f"{comp_trainable:>12,} trainable | {comp_size:>7.2f} MB")

    print("="*80 + "\n")


def _prepare_tensor(tensor: torch.Tensor, device: torch.device) -> Optional[torch.Tensor]:
    """Move tensor to device if it has data; otherwise return None."""
    if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        return None
    return tensor.to(device=device, dtype=torch.float32, non_blocking=True)


def _interpolate_robot_state_pos_enc(state_dict: Dict, model: QwenVLAUnified) -> Dict:
    """Handle positional encoding length mismatches."""
    key = "robot_state_encoder.pos_encoder"
    if key not in state_dict or not hasattr(model, "robot_state_encoder"):
        return state_dict

    pretrained = state_dict[key]
    current = model.robot_state_encoder.pos_encoder

    if pretrained.shape[1] == current.shape[1]:
        return state_dict

    # Resize along temporal dimension
    resized = torch.nn.functional.interpolate(
        pretrained.permute(0, 2, 1),
        size=current.shape[1],
        mode="linear",
        align_corners=True,
    ).permute(0, 2, 1)

    state_dict[key] = resized
    return state_dict


def compute_metrics(gt: np.ndarray, pred: np.ndarray, threshold_mm: float = 5.0) -> Dict:
    """
    Compute comprehensive metrics for action prediction.

    Args:
        gt: Ground truth actions (N, H, D) or (N, D)
        pred: Predicted actions (N, H, D) or (N, D)
        threshold_mm: Success threshold in mm for position error

    Returns:
        Dictionary with all metrics
    """
    # Flatten if needed
    if gt.ndim == 3:
        gt = gt.reshape(-1, gt.shape[-1])
        pred = pred.reshape(-1, pred.shape[-1])

    diff = pred - gt
    abs_diff = np.abs(diff)

    # Overall metrics
    mse = np.mean(np.square(diff))
    rmse = np.sqrt(mse)
    mae = np.mean(abs_diff)

    # Per-dimension metrics
    mse_per_dim = np.mean(np.square(diff), axis=0)
    rmse_per_dim = np.sqrt(mse_per_dim)
    mae_per_dim = np.mean(abs_diff, axis=0)

    # Position metrics (convert to mm assuming input is in meters)
    pos_error = np.linalg.norm(diff[:, POSITION_DIMS], axis=1) * 1000  # to mm
    pos_rmse_mm = np.sqrt(np.mean(np.square(pos_error)))
    pos_mae_mm = np.mean(pos_error)
    pos_max_mm = np.max(pos_error)

    # Rotation metrics (convert to degrees assuming input is in radians)
    rot_error = np.linalg.norm(diff[:, ROTATION_DIMS], axis=1) * 180 / np.pi  # to degrees
    rot_rmse_deg = np.sqrt(np.mean(np.square(rot_error)))
    rot_mae_deg = np.mean(rot_error)
    rot_max_deg = np.max(rot_error)

    # Gripper accuracy (binary classification)
    if gt.shape[1] > GRIPPER_DIM:
        gt_gripper = (gt[:, GRIPPER_DIM] > 0.5).astype(int)
        pred_gripper = (pred[:, GRIPPER_DIM] > 0.5).astype(int)
        gripper_accuracy = np.mean(gt_gripper == pred_gripper)
    else:
        gripper_accuracy = None

    # Success rate (position error < threshold)
    success_rate = np.mean(pos_error < threshold_mm)

    return {
        "overall": {
            "rmse": float(rmse),
            "mae": float(mae),
            "mse": float(mse),
        },
        "position": {
            "rmse_mm": float(pos_rmse_mm),
            "mae_mm": float(pos_mae_mm),
            "max_error_mm": float(pos_max_mm),
            "success_rate": float(success_rate),
        },
        "rotation": {
            "rmse_deg": float(rot_rmse_deg),
            "mae_deg": float(rot_mae_deg),
            "max_error_deg": float(rot_max_deg),
        },
        "gripper": {
            "accuracy": float(gripper_accuracy) if gripper_accuracy is not None else None,
        },
        "per_dimension": {
            ACTION_LABELS[i]: {
                "rmse": float(rmse_per_dim[i]),
                "mae": float(mae_per_dim[i]),
            }
            for i in range(min(len(ACTION_LABELS), len(rmse_per_dim)))
        },
    }


def evaluate_episode(
    model: QwenVLAUnified,
    dataset: UnifiedVLADataset,
    device: torch.device,
    batch_size: int = 4,
    num_workers: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[Dict], Dict[str, float]]:
    """
    Evaluate model on a single episode dataset.

    Returns:
        gt_actions: (N, H, D) ground truth actions
        pred_actions: (N, H, D) predicted actions
        per_sample_data: List of per-sample information
        timing_stats: Dictionary with inference timing statistics
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=unified_collate_fn,
        pin_memory=device.type.startswith("cuda"),
    )

    all_gt = []
    all_pred = []
    per_sample_data = []

    # Timing statistics
    inference_times = []
    total_samples = 0

    model.eval()
    with torch.inference_mode():
        for batch in dataloader:
            sensor_data = _prepare_tensor(batch.get("sensor_data"), device)
            robot_states = _prepare_tensor(batch.get("robot_states"), device)

            vl_cache_metadata = {
                "dataset_names": batch.get("episode_ids"),
                "vlm_indices": batch.get("vlm_indices"),
                "prompt_hashes": batch.get("prompt_hash"),
            }

            # Measure inference time
            if device.type.startswith("cuda"):
                torch.cuda.synchronize()

            start_time = time.time()

            outputs, _, _ = model(
                text_inputs=batch["instruction"],
                image_inputs=batch["images"],
                actions=None,
                sensor_data=sensor_data,
                robot_states=robot_states,
                cache_keys=batch["cache_keys"],
                vl_cache_tokens=batch.get("vl_cache"),
                vl_cache_metadata=vl_cache_metadata,
            )

            if device.type.startswith("cuda"):
                torch.cuda.synchronize()

            end_time = time.time()
            batch_inference_time = end_time - start_time
            batch_size_actual = outputs.shape[0]

            inference_times.append(batch_inference_time)
            total_samples += batch_size_actual

            pred_np = outputs.detach().cpu().float().numpy()
            gt_np = batch["actions"].cpu().float().numpy()

            all_pred.append(pred_np)
            all_gt.append(gt_np)

            # Store per-sample info
            for b in range(batch_size_actual):
                per_sample_data.append({
                    "cache_key": batch["cache_keys"][b],
                    "episode_id": batch["episode_ids"][b],
                    "vlm_idx": int(batch["vlm_indices"][b]),
                    "gt_actions": gt_np[b].tolist(),
                    "pred_actions": pred_np[b].tolist(),
                })

    gt_actions = np.concatenate(all_gt, axis=0)
    pred_actions = np.concatenate(all_pred, axis=0)

    # Compute timing statistics
    total_time = sum(inference_times)
    timing_stats = {
        "total_inference_time_sec": float(total_time),
        "avg_time_per_sample_ms": float(total_time * 1000 / total_samples) if total_samples > 0 else 0.0,
        "avg_time_per_batch_ms": float(np.mean(inference_times) * 1000) if inference_times else 0.0,
        "total_samples": total_samples,
    }

    return gt_actions, pred_actions, per_sample_data, timing_stats


def find_episodes(dataset_paths: List[str], val_split: float = 0.0, sample_episodes: int = None) -> List[Path]:
    """
    Find all episode directories matching the given patterns.

    Args:
        dataset_paths: List of glob patterns for dataset paths
        val_split: If > 0, take last X% of episodes as validation set
        sample_episodes: If set, randomly sample N episodes

    Returns:
        List of episode directory paths
    """
    all_episodes = []

    for pattern in dataset_paths:
        # Check if pattern contains wildcards
        if '*' in pattern:
            matched_dirs = glob.glob(pattern)
            for matched_dir in matched_dirs:
                path = Path(matched_dir)
                if path.is_dir():
                    # Check if this is an episode dir or contains episode dirs
                    if (path / "metadata.json").exists():
                        all_episodes.append(path)
                    else:
                        # Look for episode subdirectories
                        for subdir in path.iterdir():
                            if subdir.is_dir() and (subdir / "metadata.json").exists():
                                all_episodes.append(subdir)
        else:
            path = Path(pattern)
            if path.is_dir():
                if (path / "metadata.json").exists():
                    all_episodes.append(path)
                else:
                    for subdir in path.iterdir():
                        if subdir.is_dir() and (subdir / "metadata.json").exists():
                            all_episodes.append(subdir)

    # Remove duplicates
    all_episodes = list(set(all_episodes))
    all_episodes.sort()

    # Apply validation split
    if val_split > 0:
        split_idx = int(len(all_episodes) * (1 - val_split))
        all_episodes = all_episodes[split_idx:]

    # Sample episodes
    if sample_episodes is not None and sample_episodes < len(all_episodes):
        import random
        random.seed(42)
        all_episodes = random.sample(all_episodes, sample_episodes)

    return all_episodes


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Flow Matching VLA Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Evaluation targets
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--dataset-paths", type=str, nargs='+', required=True,
                        help="List of dataset paths or glob patterns")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation results")

    # Episode selection
    parser.add_argument("--val-split", type=float, default=0.0,
                        help="Use last X%% of episodes as validation set")
    parser.add_argument("--sample-episodes", type=int, default=None,
                        help="Randomly sample N episodes for evaluation")

    # Dataset/dataloader args
    parser.add_argument("--cache-root", type=str, default="/home/najo/NAS/VLA/dataset/cache/qwen_vl_features")
    parser.add_argument("--prompt-hash-override", type=str, default=None)
    parser.add_argument("--vlm-reuse-count", type=int, default=3)
    parser.add_argument("--sensor-window-size", type=int, default=65)
    parser.add_argument("--robot-window-size", type=int, default=100)
    parser.add_argument("--action-expert-hz", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=7)

    # Model args
    parser.add_argument("--vl-model-name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--finetune-vl", type=str, default="none", choices=["none", "lora", "full"])
    parser.add_argument("--sensor-encoder-type", type=str, default="force_aware")
    parser.add_argument("--sensor-hidden-dim", type=int, default=512)
    parser.add_argument("--sensor-output-dim", type=int, default=1024)
    parser.add_argument("--sensor-transformer-dim", type=int, default=None)
    parser.add_argument("--robot-state-output-dim", type=int, default=1024)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--flow-solver", type=str, default="euler")
    parser.add_argument("--image-resize-height", type=int, default=360)
    parser.add_argument("--image-resize-width", type=int, default=640)

    # Cache options
    parser.add_argument("--use-cache", action="store_true", default=True)
    parser.add_argument("--no-cache", dest="use_cache", action="store_false")
    parser.add_argument("--cache-only-mode", action="store_true")

    # Evaluation options
    parser.add_argument("--threshold-mm", type=float, default=5.0,
                        help="Success threshold for position error (mm)")
    parser.add_argument("--disable-sensor", action="store_true")
    parser.add_argument("--disable-robot-state", action="store_true")
    parser.add_argument("--view-indices", type=int, nargs='+', default=None,
                        help="Specific view indices to use (e.g., --view-indices 1 2 3)")

    # Logging
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="wandb project name for logging")
    parser.add_argument("--wandb-name", type=str, default=None,
                        help="wandb run name")

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Initialize wandb if requested
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"eval_{checkpoint_path.stem}_{time.strftime('%m%d_%H%M')}",
            config=vars(args),
        )

    # Find episodes to evaluate
    print("🔍 Finding episodes...")
    episodes = find_episodes(args.dataset_paths, args.val_split, args.sample_episodes)

    if len(episodes) == 0:
        raise RuntimeError(f"No episodes found matching: {args.dataset_paths}")

    print(f"✅ Found {len(episodes)} episodes to evaluate")
    if args.debug:
        for ep in episodes[:5]:
            print(f"   - {ep}")
        if len(episodes) > 5:
            print(f"   ... and {len(episodes) - 5} more")

    # Build model
    print("⏳ Loading model...")
    external_cache_root = args.cache_root if args.use_cache else None
    model = QwenVLAUnified(
        model_type="flow_matching",
        vl_model_name=args.vl_model_name,
        action_dim=args.action_dim,
        horizon=args.horizon,
        sensor_enabled=not args.disable_sensor,
        sensor_encoder_type=args.sensor_encoder_type,
        sensor_input_channels=1026,
        sensor_temporal_length=args.sensor_window_size,
        sensor_hidden_dim=args.sensor_hidden_dim,
        sensor_output_dim=args.sensor_output_dim,
        sensor_transformer_dim=args.sensor_transformer_dim,
        robot_state_enabled=not args.disable_robot_state,
        robot_state_temporal_length=args.robot_window_size,
        robot_state_output_dim=args.robot_state_output_dim,
        finetune_vl=args.finetune_vl,
        flow_steps=args.flow_steps,
        flow_solver=args.flow_solver,
        cache_dir=args.cache_root,
        external_cache_root=external_cache_root,
        image_resize_height=args.image_resize_height,
        image_resize_width=args.image_resize_width,
        cache_only_mode=args.cache_only_mode,
    ).to(device)

    # Load checkpoint
    print(f"📦 Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    state_dict = _interpolate_robot_state_pos_enc(state_dict, model)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys and args.debug:
        print(f"   ⚠️ Missing keys: {len(missing_keys)}")
    if unexpected_keys and args.debug:
        print(f"   ⚠️ Unexpected keys: {len(unexpected_keys)}")

    model.eval()
    print("✅ Model loaded successfully")

    # Calculate and print model information
    model_info = get_model_info(model)
    print_model_info(model_info)

    # Evaluate each episode
    all_episode_results = []
    all_gt_concatenated = []
    all_pred_concatenated = []

    for ep_idx, episode_path in enumerate(tqdm(episodes, desc="Evaluating episodes")):
        try:
            # Create dataset for this episode
            dataset = UnifiedVLADataset(
                data_dir=str(episode_path),
                format="auto",
                horizon=args.horizon,
                vlm_reuse_count=args.vlm_reuse_count,
                sensor_window_size=args.sensor_window_size,
                robot_window_size=args.robot_window_size,
                action_expert_hz=args.action_expert_hz,
                cache_root=args.cache_root,
                use_cache=args.use_cache,
                prompt_hash_override=args.prompt_hash_override,
                filter_by_cache=False,
                views_to_use=args.view_indices,
            )

            if len(dataset) == 0:
                print(f"⚠️ Skipping {episode_path.name}: no samples")
                continue

            # Evaluate
            gt_actions, pred_actions, per_sample_data, timing_stats = evaluate_episode(
                model, dataset, device, args.batch_size, args.num_workers
            )

            # Compute metrics
            metrics = compute_metrics(gt_actions, pred_actions, args.threshold_mm)

            # Store results
            episode_result = {
                "episode_path": str(episode_path),
                "episode_name": episode_path.name,
                "num_samples": len(dataset),
                "metrics": metrics,
                "timing": timing_stats,
                "samples": per_sample_data,
            }
            all_episode_results.append(episode_result)

            # Accumulate for overall metrics
            all_gt_concatenated.append(gt_actions)
            all_pred_concatenated.append(pred_actions)

            # Log to wandb
            if args.wandb_project:
                wandb.log({
                    f"episode/{episode_path.name}/position_rmse_mm": metrics["position"]["rmse_mm"],
                    f"episode/{episode_path.name}/rotation_rmse_deg": metrics["rotation"]["rmse_deg"],
                    f"episode/{episode_path.name}/success_rate": metrics["position"]["success_rate"],
                    f"episode/{episode_path.name}/gripper_accuracy": metrics["gripper"]["accuracy"] or 0.0,
                })

        except Exception as e:
            print(f"❌ Error evaluating {episode_path.name}: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            continue

    # Compute overall metrics
    print("\n📊 Computing overall metrics...")
    overall_gt = np.concatenate(all_gt_concatenated, axis=0)
    overall_pred = np.concatenate(all_pred_concatenated, axis=0)
    overall_metrics = compute_metrics(overall_gt, overall_pred, args.threshold_mm)

    # Compute overall timing statistics
    total_inference_time = sum(ep["timing"]["total_inference_time_sec"] for ep in all_episode_results)
    total_samples = sum(ep["timing"]["total_samples"] for ep in all_episode_results)
    overall_timing = {
        "total_inference_time_sec": float(total_inference_time),
        "avg_time_per_sample_ms": float(total_inference_time * 1000 / total_samples) if total_samples > 0 else 0.0,
        "total_samples": total_samples,
    }

    # Save results
    results = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_meta": {
            "epoch": ckpt.get("epoch"),
            "val_loss": ckpt.get("val_loss"),
            "best_val_loss": ckpt.get("best_val_loss"),
        },
        "model_info": model_info,
        "evaluation_config": {
            "num_episodes": len(episodes),
            "threshold_mm": args.threshold_mm,
            "horizon": args.horizon,
            "action_dim": args.action_dim,
            "view_indices": args.view_indices,
            "disable_sensor": args.disable_sensor,
            "disable_robot_state": args.disable_robot_state,
        },
        "overall_metrics": overall_metrics,
        "overall_timing": overall_timing,
        "episodes": all_episode_results,
    }

    results_path = output_dir / f"evaluation_results_{checkpoint_path.stem}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Evaluation complete! Results saved to: {results_path}")
    print("\n" + "="*80)
    print("📈 OVERALL METRICS")
    print("="*80)
    print(f"Position RMSE:     {overall_metrics['position']['rmse_mm']:.3f} mm")
    print(f"Position MAE:      {overall_metrics['position']['mae_mm']:.3f} mm")
    print(f"Position Max Err:  {overall_metrics['position']['max_error_mm']:.3f} mm")
    print(f"Success Rate:      {overall_metrics['position']['success_rate']*100:.2f}% (< {args.threshold_mm} mm)")
    print(f"\nRotation RMSE:     {overall_metrics['rotation']['rmse_deg']:.3f} deg")
    print(f"Rotation MAE:      {overall_metrics['rotation']['mae_deg']:.3f} deg")
    print(f"Rotation Max Err:  {overall_metrics['rotation']['max_error_deg']:.3f} deg")
    if overall_metrics['gripper']['accuracy'] is not None:
        print(f"\nGripper Accuracy:  {overall_metrics['gripper']['accuracy']*100:.2f}%")
    print(f"\n⏱️  Inference Time:   {overall_timing['avg_time_per_sample_ms']:.2f} ms/sample")
    print(f"   Total Samples:    {overall_timing['total_samples']}")
    print(f"   Total Time:       {overall_timing['total_inference_time_sec']:.2f} sec")
    print("\n" + "="*80)

    # Log overall metrics to wandb
    if args.wandb_project:
        wandb_log_dict = {
            "overall/position_rmse_mm": overall_metrics["position"]["rmse_mm"],
            "overall/position_mae_mm": overall_metrics["position"]["mae_mm"],
            "overall/position_success_rate": overall_metrics["position"]["success_rate"],
            "overall/rotation_rmse_deg": overall_metrics["rotation"]["rmse_deg"],
            "overall/rotation_mae_deg": overall_metrics["rotation"]["mae_deg"],
            "overall/gripper_accuracy": overall_metrics["gripper"]["accuracy"] or 0.0,
            "overall/avg_inference_time_ms": overall_timing["avg_time_per_sample_ms"],
            "overall/total_inference_time_sec": overall_timing["total_inference_time_sec"],
            "model/total_params_M": model_info["total_params"] / 1e6,
            "model/trainable_params_M": model_info["trainable_params"] / 1e6,
            "model/size_mb": model_info["model_size_mb"],
        }

        # Log component-level parameters
        for comp_name, comp_info in model_info["component_params"].items():
            wandb_log_dict[f"model/{comp_name}_params_M"] = comp_info["total"] / 1e6
            wandb_log_dict[f"model/{comp_name}_size_mb"] = comp_info["size_mb"]

        wandb.log(wandb_log_dict)
        wandb.finish()

    return results_path


if __name__ == "__main__":
    main()
