"""
python evaluation_results/plot_first_delta_action.py \
    --results-json /home/najo/NAS/VLA/Insertion_VLAv3/evaluation_results/data_collection_20251115_012126_eval.json
    
    

Utility to visualize translation/rotation delta-action comparisons from evaluation JSON."""


import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)


TRANSLATION_IDXS = [0, 1, 2]
ROTATION_IDXS = [3, 4, 5]
DIM_LABELS = ["dx", "dy", "dz", "rx", "ry", "rz", "gripper"]


def _flatten_delta_actions(results_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with results_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    gt_chunks = []
    pred_chunks = []
    for sample in data.get("samples", []):
        gt_actions = np.asarray(sample.get("gt_delta_actions", []), dtype=float)
        pred_actions = np.asarray(sample.get("pred_delta_actions", []), dtype=float)

        if gt_actions.shape != pred_actions.shape:
            raise ValueError(
                f"GT/pred length mismatch in sample {sample.get('sample_idx')}: "
                f"{gt_actions.shape} vs {pred_actions.shape}"
            )

        if gt_actions.ndim != 2:
            raise ValueError("Delta action arrays must be 2D (horizon x action_dim)")

        gt_chunks.append(gt_actions)
        pred_chunks.append(pred_actions)

    if not gt_chunks:
        raise ValueError(f"No samples found in {results_path}")

    gt_all = np.concatenate(gt_chunks, axis=0)
    pred_all = np.concatenate(pred_chunks, axis=0)
    return gt_all, pred_all


def plot_series(x, gt, pred, title, ylabel, output_path: Path):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, gt, label="Ground Truth", linewidth=2)
    ax.plot(x, pred, label="Prediction", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_3d_translation(gt_xyz: np.ndarray, pred_xyz: np.ndarray, output_path: Path):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], label="GT", linewidth=2)
    ax.plot(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2], label="Prediction", linewidth=2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Cumulative XYZ Delta Trajectories")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot GT vs predicted delta actions (translation + rotation) from evaluation JSON."
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        required=True,
        help="Path to JSON produced by evaluate_flowmatching_episode.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_results"),
        help="Directory to store the generated figures.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Optional filename prefix for the output figures.",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gt_vals, pred_vals = _flatten_delta_actions(args.results_json)
    steps = np.arange(gt_vals.shape[0])

    prefix = args.output_prefix or args.results_json.stem

    # Translation components (dims 0-2): raw + cumulative plots
    for dim in TRANSLATION_IDXS:
        if dim >= gt_vals.shape[1]:
            continue
        label = DIM_LABELS[dim] if dim < len(DIM_LABELS) else f"dim{dim}"
        plot_series(
            steps,
            gt_vals[:, dim],
            pred_vals[:, dim],
            f"Delta {label}",
            f"Delta {label}",
            args.output_dir / f"{prefix}_delta_{label}.png",
        )
        plot_series(
            steps,
            np.cumsum(gt_vals[:, dim]),
            np.cumsum(pred_vals[:, dim]),
            f"Cumulative {label}",
            f"Cumulative {label}",
            args.output_dir / f"{prefix}_delta_{label}_cumsum.png",
        )

    # 3D cumulative trajectory (XYZ)
    if all(dim < gt_vals.shape[1] for dim in TRANSLATION_IDXS):
        gt_xyz = np.cumsum(gt_vals[:, TRANSLATION_IDXS], axis=0)
        pred_xyz = np.cumsum(pred_vals[:, TRANSLATION_IDXS], axis=0)
        plot_3d_translation(
            gt_xyz,
            pred_xyz,
            args.output_dir / f"{prefix}_delta_xyz_traj.png",
        )

    # Rotation components (dims 3-5): raw comparison plots
    for dim in ROTATION_IDXS:
        if dim >= gt_vals.shape[1]:
            continue
        label = DIM_LABELS[dim] if dim < len(DIM_LABELS) else f"dim{dim}"
        plot_series(
            steps,
            gt_vals[:, dim],
            pred_vals[:, dim],
            f"Delta {label}",
            f"Delta {label}",
            args.output_dir / f"{prefix}_delta_{label}.png",
        )

    print("Saved translation/rotation comparison plots to", args.output_dir)


if __name__ == "__main__":
    main()
