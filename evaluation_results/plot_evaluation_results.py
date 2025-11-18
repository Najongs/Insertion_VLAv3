#!/usr/bin/env python3
"""
Visualization script for comprehensive VLA evaluation results.

Creates multiple plots from EVAL_FlowMatching.py output:
- Overall error distribution histograms
- Per-dimension RMSE comparison
- Trajectory visualization (GT vs Predicted)
- Per-episode performance comparison
- Error over time analysis

Usage:
    python evaluation_results/plot_evaluation_results.py \
        --results-json evaluation_results/evaluation_results_flow_matching_best.json \
        --output-dir evaluation_results/plots
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

sns.set_style("whitegrid")

ACTION_LABELS = ["dx", "dy", "dz", "rx", "ry", "rz", "gripper"]
POSITION_DIMS = [0, 1, 2]
ROTATION_DIMS = [3, 4, 5]


def load_results(results_path: Path) -> Dict:
    """Load evaluation results JSON."""
    with open(results_path, 'r') as f:
        return json.load(f)


def extract_all_predictions(results: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract all GT and predictions from results.

    Returns:
        gt_all: (N, H, D) array
        pred_all: (N, H, D) array
    """
    gt_list = []
    pred_list = []

    for episode in results.get("episodes", []):
        for sample in episode.get("samples", []):
            gt_list.append(np.array(sample["gt_actions"]))
            pred_list.append(np.array(sample["pred_actions"]))

    if not gt_list:
        raise ValueError("No samples found in results")

    return np.array(gt_list), np.array(pred_list)


def plot_error_distribution(gt: np.ndarray, pred: np.ndarray, output_path: Path):
    """Plot error distribution histograms for each dimension."""
    # Flatten to (N*H, D)
    gt_flat = gt.reshape(-1, gt.shape[-1])
    pred_flat = pred.reshape(-1, pred.shape[-1])
    errors = pred_flat - gt_flat

    n_dims = errors.shape[1]
    n_cols = 3
    n_rows = (n_dims + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]

    for i in range(n_dims):
        ax = axes[i]
        label = ACTION_LABELS[i] if i < len(ACTION_LABELS) else f"dim_{i}"

        # Convert to appropriate units
        if i in POSITION_DIMS:
            err_data = errors[:, i] * 1000  # to mm
            unit = "mm"
        elif i in ROTATION_DIMS:
            err_data = errors[:, i] * 180 / np.pi  # to degrees
            unit = "deg"
        else:
            err_data = errors[:, i]
            unit = ""

        ax.hist(err_data, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax.set_xlabel(f"Error ({unit})" if unit else "Error")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{label} Error Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_err = np.mean(err_data)
        std_err = np.std(err_data)
        ax.text(0.05, 0.95, f"Mean: {mean_err:.3f}\nStd: {std_err:.3f}",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Hide unused subplots
    for i in range(n_dims, len(axes)):
        axes[i].axis('off')

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"✅ Saved error distribution plot: {output_path}")


def plot_per_dimension_rmse(results: Dict, output_path: Path):
    """Plot RMSE comparison across dimensions."""
    overall_metrics = results["overall_metrics"]["per_dimension"]

    dims = []
    rmse_values = []

    for dim_name, metrics in overall_metrics.items():
        dims.append(dim_name)
        rmse_values.append(metrics["rmse"])

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['blue' if i in POSITION_DIMS else 'green' if i in ROTATION_DIMS else 'orange'
              for i in range(len(dims))]

    bars = ax.bar(dims, rmse_values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel("RMSE")
    ax.set_title("Per-Dimension RMSE")
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=9)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, label='Position (dx, dy, dz)'),
        Patch(facecolor='green', alpha=0.7, label='Rotation (rx, ry, rz)'),
        Patch(facecolor='orange', alpha=0.7, label='Gripper'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"✅ Saved per-dimension RMSE plot: {output_path}")


def plot_3d_trajectories(gt: np.ndarray, pred: np.ndarray, output_path: Path, max_samples: int = 5):
    """Plot 3D cumulative trajectories for position (xyz)."""
    # Take first few episodes for visualization
    n_samples = min(max_samples, gt.shape[0])

    fig = plt.figure(figsize=(12, 10))

    for idx in range(n_samples):
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')

        # Extract position deltas
        gt_xyz = gt[idx, :, POSITION_DIMS]  # (H, 3)
        pred_xyz = pred[idx, :, POSITION_DIMS]  # (H, 3)

        # Cumulative sum to get trajectory
        gt_traj = np.cumsum(gt_xyz, axis=0) * 1000  # to mm
        pred_traj = np.cumsum(pred_xyz, axis=0) * 1000  # to mm

        ax.plot(gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2],
                'b-o', label='GT', linewidth=2, markersize=4)
        ax.plot(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2],
                'r--s', label='Pred', linewidth=2, markersize=4)

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'Sample {idx + 1}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Last subplot: overall statistics
    if n_samples < 6:
        ax = fig.add_subplot(2, 3, 6)
        ax.axis('off')

        # Compute overall position error
        gt_flat = gt[:, :, POSITION_DIMS].reshape(-1, 3)
        pred_flat = pred[:, :, POSITION_DIMS].reshape(-1, 3)
        pos_errors = np.linalg.norm(pred_flat - gt_flat, axis=1) * 1000

        stats_text = f"""
Overall Position Statistics:

Mean Error: {np.mean(pos_errors):.3f} mm
Median Error: {np.median(pos_errors):.3f} mm
Std Error: {np.std(pos_errors):.3f} mm
Max Error: {np.max(pos_errors):.3f} mm
Min Error: {np.min(pos_errors):.3f} mm

Total Samples: {gt.shape[0]}
Horizon: {gt.shape[1]}
        """
        ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center')

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"✅ Saved 3D trajectory plot: {output_path}")


def plot_per_episode_metrics(results: Dict, output_path: Path):
    """Plot per-episode performance comparison."""
    episodes = results.get("episodes", [])

    if len(episodes) == 0:
        print("⚠️ No episodes found, skipping per-episode plot")
        return

    episode_names = []
    pos_rmse = []
    rot_rmse = []
    success_rates = []

    for ep in episodes:
        episode_names.append(ep["episode_name"][:20])  # Truncate long names
        pos_rmse.append(ep["metrics"]["position"]["rmse_mm"])
        rot_rmse.append(ep["metrics"]["rotation"]["rmse_deg"])
        success_rates.append(ep["metrics"]["position"]["success_rate"] * 100)

    fig, axes = plt.subplots(3, 1, figsize=(max(10, len(episodes) * 0.5), 12))

    # Position RMSE
    axes[0].bar(episode_names, pos_rmse, color='blue', alpha=0.7, edgecolor='black')
    axes[0].set_ylabel("Position RMSE (mm)")
    axes[0].set_title("Position Error per Episode")
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Rotation RMSE
    axes[1].bar(episode_names, rot_rmse, color='green', alpha=0.7, edgecolor='black')
    axes[1].set_ylabel("Rotation RMSE (deg)")
    axes[1].set_title("Rotation Error per Episode")
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')

    # Success Rate
    axes[2].bar(episode_names, success_rates, color='orange', alpha=0.7, edgecolor='black')
    axes[2].set_ylabel("Success Rate (%)")
    axes[2].set_title("Success Rate per Episode")
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].axhline(y=100, color='red', linestyle='--', linewidth=2, label='100%')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"✅ Saved per-episode metrics plot: {output_path}")


def plot_temporal_error(gt: np.ndarray, pred: np.ndarray, output_path: Path):
    """Plot error over horizon (temporal analysis)."""
    # gt, pred: (N, H, D)
    errors = pred - gt

    # Position error per horizon step
    pos_errors = np.linalg.norm(errors[:, :, POSITION_DIMS], axis=2) * 1000  # (N, H) in mm

    horizon = gt.shape[1]
    mean_error = np.mean(pos_errors, axis=0)
    std_error = np.std(pos_errors, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))

    steps = np.arange(1, horizon + 1)
    ax.plot(steps, mean_error, 'b-o', linewidth=2, markersize=8, label='Mean Error')
    ax.fill_between(steps, mean_error - std_error, mean_error + std_error,
                     alpha=0.3, label='±1 Std')

    ax.set_xlabel("Horizon Step")
    ax.set_ylabel("Position Error (mm)")
    ax.set_title("Position Error Over Prediction Horizon")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add value labels
    for i, (step, err) in enumerate(zip(steps, mean_error)):
        ax.text(step, err, f'{err:.2f}', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"✅ Saved temporal error plot: {output_path}")


def plot_summary_report(results: Dict, output_path: Path):
    """Create a summary report figure with key metrics."""
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle("VLA Model Evaluation Summary", fontsize=16, fontweight='bold')

    # Subplot 1: Overall metrics table
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.axis('off')

    overall = results["overall_metrics"]
    summary_text = f"""
OVERALL METRICS

Position:
  RMSE: {overall['position']['rmse_mm']:.3f} mm
  MAE:  {overall['position']['mae_mm']:.3f} mm
  Max:  {overall['position']['max_error_mm']:.3f} mm
  Success Rate: {overall['position']['success_rate']*100:.2f}%

Rotation:
  RMSE: {overall['rotation']['rmse_deg']:.3f} deg
  MAE:  {overall['rotation']['mae_deg']:.3f} deg
  Max:  {overall['rotation']['max_error_deg']:.3f} deg

Gripper:
  Accuracy: {overall['gripper']['accuracy']*100:.2f}%
    """ if overall['gripper']['accuracy'] is not None else f"""
OVERALL METRICS

Position:
  RMSE: {overall['position']['rmse_mm']:.3f} mm
  MAE:  {overall['position']['mae_mm']:.3f} mm
  Max:  {overall['position']['max_error_mm']:.3f} mm
  Success Rate: {overall['position']['success_rate']*100:.2f}%

Rotation:
  RMSE: {overall['rotation']['rmse_deg']:.3f} deg
  MAE:  {overall['rotation']['mae_deg']:.3f} deg
  Max:  {overall['rotation']['max_error_deg']:.3f} deg
    """

    ax1.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center')

    # Subplot 2: Per-dimension RMSE
    ax2 = fig.add_subplot(2, 2, 2)
    dims = []
    rmse_vals = []
    for dim_name, metrics in overall["per_dimension"].items():
        dims.append(dim_name)
        rmse_vals.append(metrics["rmse"])

    colors = ['blue' if i in POSITION_DIMS else 'green' if i in ROTATION_DIMS else 'orange'
              for i in range(len(dims))]
    ax2.bar(dims, rmse_vals, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel("RMSE")
    ax2.set_title("Per-Dimension RMSE")
    ax2.grid(True, alpha=0.3, axis='y')

    # Subplot 3: Episode count and config
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.axis('off')

    config = results["evaluation_config"]
    ckpt_meta = results["checkpoint_meta"]

    config_text = f"""
EVALUATION CONFIG

Checkpoint: {Path(results['checkpoint']).name}
Epoch: {ckpt_meta.get('epoch', 'N/A')}
Val Loss: {ckpt_meta.get('val_loss', 'N/A')}

Episodes Evaluated: {config['num_episodes']}
Success Threshold: {config['threshold_mm']} mm
Horizon: {config['horizon']}
Action Dim: {config['action_dim']}
    """

    ax3.text(0.1, 0.5, config_text, fontsize=11, family='monospace',
             verticalalignment='center')

    # Subplot 4: Quick stats
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    total_samples = sum(ep["num_samples"] for ep in results["episodes"])
    avg_samples_per_ep = total_samples / config["num_episodes"] if config["num_episodes"] > 0 else 0

    stats_text = f"""
DATASET STATISTICS

Total Samples: {total_samples}
Avg Samples/Episode: {avg_samples_per_ep:.1f}

Total Predictions: {total_samples * config['horizon']}
    """

    ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"✅ Saved summary report: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize VLA evaluation results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--results-json", type=Path, required=True,
                        help="Path to evaluation results JSON from EVAL_FlowMatching.py")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory to save plots (default: same as JSON)")
    parser.add_argument("--max-trajectory-samples", type=int, default=5,
                        help="Max number of trajectory samples to plot")

    args = parser.parse_args()

    if not args.results_json.exists():
        raise FileNotFoundError(f"Results JSON not found: {args.results_json}")

    # Setup output directory
    if args.output_dir is None:
        args.output_dir = args.results_json.parent / "plots"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📊 Loading results from: {args.results_json}")
    results = load_results(args.results_json)

    print(f"📈 Generating plots...")

    # Extract all predictions
    gt_all, pred_all = extract_all_predictions(results)
    print(f"   Loaded {gt_all.shape[0]} samples with horizon {gt_all.shape[1]}")

    # Generate plots
    prefix = args.results_json.stem

    # 1. Error distribution
    plot_error_distribution(gt_all, pred_all, args.output_dir / f"{prefix}_error_distribution.png")

    # 2. Per-dimension RMSE
    plot_per_dimension_rmse(results, args.output_dir / f"{prefix}_per_dim_rmse.png")

    # 3. 3D trajectories
    plot_3d_trajectories(gt_all, pred_all, args.output_dir / f"{prefix}_3d_trajectories.png",
                         max_samples=args.max_trajectory_samples)

    # 4. Per-episode metrics
    plot_per_episode_metrics(results, args.output_dir / f"{prefix}_per_episode_metrics.png")

    # 5. Temporal error
    plot_temporal_error(gt_all, pred_all, args.output_dir / f"{prefix}_temporal_error.png")

    # 6. Summary report
    plot_summary_report(results, args.output_dir / f"{prefix}_summary_report.png")

    print(f"\n✅ All plots saved to: {args.output_dir}")
    print(f"   Generated {6} visualization files")


if __name__ == "__main__":
    main()
