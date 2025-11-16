#!/usr/bin/env python3
"""
Evaluate a trained Flow Matching VLA checkpoint on a single episode.

The script loads the specified checkpoint, runs inference over every action
window of the provided episode, and stores the following artifacts:
  • Per-sample ground-truth and predicted delta actions (JSON friendly).
  • RMSE metrics (overall + per-action-dimension) for quick comparison.
  
python3 evaluate_flowmatching_episode.py \
    --episode-path /home/najo/NAS/VLA/dataset/New_dataset4/Eye_trocar/data_collection_20251115_001852 \
    --checkpoint checkpoints/flow_matching_latest.pt \
    --output evaluation_results/data_collection_20251115_012126_eval.json \
    --sensor-hidden-dim 128 \
    --sensor-transformer-dim 256 \
    --batch-size 4 --num-workers 0
    
python evaluation_results/plot_first_delta_action.py \
    --results-json /home/najo/NAS/VLA/Insertion_VLAv3/evaluation_results/data_collection_20251115_012126_eval.json
  
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.unified_model import QwenVLAUnified
from vla_datasets.unified_dataset import UnifiedVLADataset, unified_collate_fn


ACTION_LABELS = ["dx", "dy", "dz", "rx", "ry", "rz", "gripper"]


def _prepare_tensor(tensor: torch.Tensor, device: torch.device) -> Optional[torch.Tensor]:
    """Move tensor to device if it has data; otherwise return None."""
    if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        return None
    return tensor.to(device=device, dtype=torch.float32, non_blocking=True)


def _interpolate_robot_state_pos_enc(state_dict: Dict[str, torch.Tensor], model: QwenVLAUnified) -> Dict[str, torch.Tensor]:
    """
    Handle positional encoding length mismatches between checkpoint and current model.
    """
    key = "robot_state_encoder.pos_encoder"
    if key not in state_dict or not hasattr(model, "robot_state_encoder"):
        return state_dict

    pretrained = state_dict[key]
    current = model.robot_state_encoder.pos_encoder

    if pretrained.shape[1] == current.shape[1]:
        return state_dict

    # Resize along the temporal dimension using linear interpolation.
    resized = torch.nn.functional.interpolate(
        pretrained.permute(0, 2, 1),
        size=current.shape[1],
        mode="linear",
        align_corners=True,
    ).permute(0, 2, 1)

    state_dict[key] = resized
    return state_dict


def _flatten_for_metrics(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate a list of [N, H, D] arrays into [-1, D] form."""
    if not arrays:
        return np.zeros((0, len(ACTION_LABELS)), dtype=np.float32)
    stacked = np.concatenate(arrays, axis=0)
    return stacked.reshape(-1, stacked.shape[-1])


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a Flow Matching checkpoint on a single episode and save GT/pred JSON + RMSE metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--episode-path", type=str, required=True, help="Path to the episode directory (contains metadata.json, sensor files, images).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained Flow Matching checkpoint (.pt).")
    parser.add_argument("--output", type=str, default=None, help="Path to the JSON file that will store GT vs prediction data.")

    # Dataset / dataloader args
    parser.add_argument("--cache-root", type=str, default="/home/najo/NAS/VLA/dataset/cache/qwen_vl_features", help="VL feature cache directory.")
    parser.add_argument("--prompt-hash-override", type=str, default=None, help="Override prompt hash when cache was built with a different instruction.")
    parser.add_argument("--vlm-reuse-count", type=int, default=3, help="Reuse count that was used during cache building/training.")
    parser.add_argument("--sensor-window-size", type=int, default=65, help="Sensor window size fed to the model.")
    parser.add_argument("--robot-window-size", type=int, default=100, help="Robot state window size fed to the model.")
    parser.add_argument("--action-expert-hz", type=int, default=10, help="Action expert frequency (should match training).")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=7)

    parser.add_argument("--disable-cache", dest="use_cache", action="store_false", help="Force raw image encoding by disabling VL cache usage.")
    parser.set_defaults(use_cache=True)
    parser.add_argument("--cache-only-mode", action="store_true", help="Skip loading the VL backbone and rely solely on cached tokens.")
    parser.add_argument("--disable-cache-backfill", action="store_true", help="Disable auto backfilling of missing cache entries during evaluation.")

    # Model args
    parser.add_argument("--vl-model-name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--finetune-vl", type=str, default="none", choices=["none", "lora", "full"])
    parser.add_argument("--sensor-encoder-type", type=str, default="force_aware", choices=["default", "force_aware"])
    parser.add_argument("--sensor-hidden-dim", type=int, default=512)
    parser.add_argument("--sensor-output-dim", type=int, default=1024)
    parser.add_argument("--sensor-transformer-dim", type=int, default=None)
    parser.add_argument("--robot-state-output-dim", type=int, default=1024)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--flow-solver", type=str, default="euler", choices=["euler", "heun"])
    parser.add_argument("--image-resize-height", type=int, default=None)
    parser.add_argument("--image-resize-width", type=int, default=None)
    parser.add_argument("--disable-sensor", action="store_true", help="Disable sensor encoder usage at inference.")
    parser.add_argument("--disable-robot-state", action="store_true", help="Disable robot state encoder usage at inference.")

    parser.add_argument("--device", type=str, default=None, help="Device to run evaluation on (e.g., cuda:0). Defaults to CUDA if available.")
    parser.add_argument("--debug", action="store_true", help="Print extra debugging information.")

    return parser


def build_dataset(args: argparse.Namespace) -> UnifiedVLADataset:
    return UnifiedVLADataset(
        data_dir=args.episode_path,
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
    )


def build_model(args: argparse.Namespace, device: torch.device) -> QwenVLAUnified:
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
        auto_cache_backfill=not args.disable_cache_backfill,
        image_resize_height=args.image_resize_height,
        image_resize_width=args.image_resize_width,
        cache_only_mode=args.cache_only_mode,
    )
    return model.to(device)


def load_checkpoint(model: QwenVLAUnified, ckpt_path: Path) -> Dict[str, Optional[float]]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    state_dict = _interpolate_robot_state_pos_enc(state_dict, model)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"⚠️ Missing keys while loading checkpoint: {len(missing_keys)}")
    if unexpected_keys:
        print(f"⚠️ Unexpected keys in checkpoint: {len(unexpected_keys)}")

    return {
        "epoch": ckpt.get("epoch"),
        "val_loss": ckpt.get("val_loss"),
        "best_val_loss": ckpt.get("best_val_loss"),
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
    }


def evaluate(args: argparse.Namespace) -> Path:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    episode_path = Path(args.episode_path)
    if not episode_path.exists():
        raise FileNotFoundError(f"Episode path not found: {episode_path}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dataset = build_dataset(args)
    if len(dataset) == 0:
        raise RuntimeError(f"No samples found in episode: {episode_path}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=unified_collate_fn,
        pin_memory=device.type.startswith("cuda"),
    )

    model = build_model(args, device)
    ckpt_meta = load_checkpoint(model, checkpoint_path)
    model.eval()

    all_gt: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    per_sample: List[Dict[str, object]] = []

    if args.debug:
        print(f"Evaluating on device: {device}")
        print(f"Total windows: {len(dataset)} | Batches: {len(dataloader)}")

    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", unit="batch")):
            sensor_data = _prepare_tensor(batch["sensor_data"], device)
            robot_states = _prepare_tensor(batch["robot_states"], device)

            vl_cache_metadata = {
                "dataset_names": batch.get("episode_ids"),
                "vlm_indices": batch.get("vlm_indices"),
                "prompt_hashes": batch.get("prompt_hash"),
            }

            outputs, _, _ = model(
                text_inputs=batch["instruction"],
                image_inputs=batch["images"],
                actions=None,
                sensor_data=sensor_data if not args.disable_sensor else None,
                robot_states=robot_states if not args.disable_robot_state else None,
                cache_keys=batch["cache_keys"],
                vl_cache_tokens=batch.get("vl_cache"),
                vl_cache_metadata=vl_cache_metadata,
            )

            pred_np = outputs.detach().to("cpu").float().numpy()
            gt_np = batch["actions"].to("cpu").float().numpy()

            all_pred.append(pred_np)
            all_gt.append(gt_np)

            batch_size = pred_np.shape[0]
            for b in range(batch_size):
                per_sample.append({
                    "sample_idx": len(per_sample),
                    "cache_key": batch["cache_keys"][b],
                    "episode_id": batch["episode_ids"][b],
                    "vlm_idx": int(batch["vlm_indices"][b]),
                    "gt_delta_actions": gt_np[b].tolist(),
                    "pred_delta_actions": pred_np[b].tolist(),
                })

    gt_flat = _flatten_for_metrics(all_gt)
    pred_flat = _flatten_for_metrics(all_pred)

    if gt_flat.shape[0] == 0:
        raise RuntimeError("No data collected for metric computation.")

    diff = pred_flat - gt_flat
    mse_per_dim = np.mean(np.square(diff), axis=0)
    rmse_per_dim = np.sqrt(mse_per_dim)
    overall_rmse = float(np.sqrt(np.mean(np.square(diff))))

    dim_labels = ACTION_LABELS[:gt_flat.shape[1]] if gt_flat.shape[1] <= len(ACTION_LABELS) else [f"dim_{i}" for i in range(gt_flat.shape[1])]
    rmse_dict = {label: float(rmse_per_dim[i]) for i, label in enumerate(dim_labels)}

    output_path = Path(args.output) if args.output else Path("evaluation_results") / f"{episode_path.name}_flow_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "episode_path": str(episode_path),
        "episode_id": dataset.data_dir.name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_meta": {
            "epoch": ckpt_meta.get("epoch"),
            "val_loss": ckpt_meta.get("val_loss"),
            "best_val_loss": ckpt_meta.get("best_val_loss"),
            "missing_keys": ckpt_meta.get("missing_keys"),
            "unexpected_keys": ckpt_meta.get("unexpected_keys"),
        },
        "horizon": args.horizon,
        "action_dim": args.action_dim,
        "num_windows": len(per_sample),
        "rmse_overall": overall_rmse,
        "rmse_per_dimension": rmse_dict,
        "samples": per_sample,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Evaluation complete. JSON saved to: {output_path}")
    print(f"   Overall RMSE: {overall_rmse:.6f}")
    for label, value in rmse_dict.items():
        print(f"   RMSE[{label}]: {value:.6f}")

    return output_path


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
