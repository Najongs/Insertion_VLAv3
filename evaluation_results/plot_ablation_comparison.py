#!/usr/bin/env python3
"""
Ablation Study Comparison Visualization Script

Compares evaluation results from different model configurations:
- Full model (baseline)
- Without Sensor Encoder
- Without Robot State Encoder
- Without Both Encoders (VL only)

Generates:
1. Overall metrics comparison bar charts
2. Per-dimension RMSE comparison
3. 3D Trajectory comparison (GT vs Predicted for each config)
4. Relative performance analysis (% change from baseline)
5. Summary table with all metrics

Usage:
    python evaluation_results/plot_ablation_comparison.py \
        --full-model evaluation_results/ablation_study/full_model/evaluation_results_*.json \
        --wo-sensor evaluation_results/ablation_study/wo_sensor/evaluation_results_*.json \
        --wo-robot-state evaluation_results/ablation_study/wo_robot_state/evaluation_results_*.json \
        --wo-both evaluation_results/ablation_study/wo_both/evaluation_results_*.json \
        --output-dir evaluation_results/ablation_study/comparison_plots \
        --max-trajectory-samples 3
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import numpy as np


ACTION_LABELS = ["dx", "dy", "dz", "rx", "ry", "rz", "gripper"]
POSITION_DIMS = [0, 1, 2]  # x, y, z

# Color scheme for different configurations
COLORS = {
    "full_model": "#2E7D32",       # Green - best performance
    "wo_sensor": "#F57C00",        # Orange
    "wo_robot_state": "#1976D2",   # Blue
    "wo_both": "#C62828",          # Red - worst performance
}

CONFIG_LABELS = {
    "full_model": "Full Model",
    "wo_sensor": "w/o Sensor",
    "wo_robot_state": "w/o Robot State",
    "wo_both": "w/o Both",
}


def load_results(json_path: Path) -> Dict:
    """Load evaluation results from JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def extract_sample_trajectories(results: Dict, max_samples: int = 3) -> tuple:
    """Extract sample trajectories from evaluation results."""
    all_gt = []
    all_pred = []

    for episode in results["episodes"]:
        for sample in episode["samples"][:max_samples]:
            gt = np.array(sample["gt_actions"])  # (H, D)
            pred = np.array(sample["pred_actions"])  # (H, D)
            all_gt.append(gt)
            all_pred.append(pred)

            if len(all_gt) >= max_samples:
                break
        if len(all_gt) >= max_samples:
            break

    return np.array(all_gt), np.array(all_pred)


def plot_overall_metrics_comparison(results: Dict[str, Dict], output_path: Path):
    """
    Plot overall metrics comparison across all configurations.

    Metrics plotted:
    - Position RMSE (mm)
    - Rotation RMSE (deg)
    - Success Rate (%)
    - Gripper Accuracy (%)
    """
    configs = list(results.keys())
    n_configs = len(configs)

    # Extract metrics
    pos_rmse = [results[c]["overall_metrics"]["position"]["rmse_mm"] for c in configs]
    rot_rmse = [results[c]["overall_metrics"]["rotation"]["rmse_deg"] for c in configs]
    success_rate = [results[c]["overall_metrics"]["position"]["success_rate"] * 100 for c in configs]
    gripper_acc = [
        (results[c]["overall_metrics"]["gripper"]["accuracy"] or 0.0) * 100
        for c in configs
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Ablation Study: Overall Metrics Comparison", fontsize=16, fontweight="bold")

    x = np.arange(n_configs)
    width = 0.6

    # Position RMSE
    ax = axes[0, 0]
    bars = ax.bar(x, pos_rmse, width, color=[COLORS[c] for c in configs], alpha=0.8, edgecolor='black')
    ax.set_ylabel("Position RMSE (mm)", fontsize=11, fontweight="bold")
    ax.set_title("Position Accuracy", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[c] for c in configs], rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, pos_rmse)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Rotation RMSE
    ax = axes[0, 1]
    bars = ax.bar(x, rot_rmse, width, color=[COLORS[c] for c in configs], alpha=0.8, edgecolor='black')
    ax.set_ylabel("Rotation RMSE (deg)", fontsize=11, fontweight="bold")
    ax.set_title("Rotation Accuracy", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[c] for c in configs], rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    for i, (bar, val) in enumerate(zip(bars, rot_rmse)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Success Rate
    ax = axes[1, 0]
    bars = ax.bar(x, success_rate, width, color=[COLORS[c] for c in configs], alpha=0.8, edgecolor='black')
    ax.set_ylabel("Success Rate (%)", fontsize=11, fontweight="bold")
    ax.set_title("Success Rate", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[c] for c in configs], rotation=15, ha='right')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    for i, (bar, val) in enumerate(zip(bars, success_rate)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Gripper Accuracy
    ax = axes[1, 1]
    bars = ax.bar(x, gripper_acc, width, color=[COLORS[c] for c in configs], alpha=0.8, edgecolor='black')
    ax.set_ylabel("Gripper Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("Gripper Prediction Accuracy", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[c] for c in configs], rotation=15, ha='right')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    for i, (bar, val) in enumerate(zip(bars, gripper_acc)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def plot_per_dimension_comparison(results: Dict[str, Dict], output_path: Path):
    """Plot per-dimension RMSE comparison across all configurations."""
    configs = list(results.keys())
    n_configs = len(configs)
    n_dims = len(ACTION_LABELS)

    # Extract per-dimension RMSE
    per_dim_data = []
    for config in configs:
        per_dim = results[config]["overall_metrics"]["per_dimension"]
        rmse_vals = [per_dim[label]["rmse"] for label in ACTION_LABELS]
        per_dim_data.append(rmse_vals)

    per_dim_data = np.array(per_dim_data)  # Shape: (n_configs, n_dims)

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(n_dims)
    width = 0.2

    for i, config in enumerate(configs):
        offset = (i - n_configs / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            per_dim_data[i],
            width,
            label=CONFIG_LABELS[config],
            color=COLORS[config],
            alpha=0.8,
            edgecolor='black',
        )

    ax.set_ylabel("RMSE", fontsize=12, fontweight="bold")
    ax.set_xlabel("Action Dimension", fontsize=12, fontweight="bold")
    ax.set_title("Ablation Study: Per-Dimension RMSE Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(ACTION_LABELS)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def plot_relative_performance(results: Dict[str, Dict], output_path: Path):
    """
    Plot relative performance degradation compared to full model baseline.

    Shows percentage change in key metrics when components are removed.
    Negative values indicate worse performance (higher error).
    """
    baseline = results["full_model"]
    configs = [c for c in results.keys() if c != "full_model"]

    # Metrics to compare (lower is better, so degradation is positive % increase)
    metrics = {
        "Position RMSE (mm)": lambda r: r["overall_metrics"]["position"]["rmse_mm"],
        "Rotation RMSE (deg)": lambda r: r["overall_metrics"]["rotation"]["rmse_deg"],
        "Success Rate (%)": lambda r: r["overall_metrics"]["position"]["success_rate"] * 100,
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Ablation Study: Relative Performance Change from Baseline", fontsize=14, fontweight="bold")

    for idx, (metric_name, metric_fn) in enumerate(metrics.items()):
        ax = axes[idx]

        baseline_val = metric_fn(baseline)

        pct_changes = []
        for config in configs:
            current_val = metric_fn(results[config])

            # For error metrics (RMSE), positive % means degradation
            # For success rate, negative % means degradation
            if "Success Rate" in metric_name:
                pct_change = ((current_val - baseline_val) / baseline_val) * 100
            else:
                pct_change = ((current_val - baseline_val) / baseline_val) * 100

            pct_changes.append(pct_change)

        x = np.arange(len(configs))
        colors_list = [COLORS[c] for c in configs]

        bars = ax.barh(x, pct_changes, color=colors_list, alpha=0.8, edgecolor='black')

        ax.set_yticks(x)
        ax.set_yticklabels([CONFIG_LABELS[c] for c in configs])
        ax.set_xlabel("% Change from Baseline", fontsize=10, fontweight="bold")
        ax.set_title(metric_name, fontsize=11, fontweight="bold")
        ax.axvline(0, color='black', linewidth=1.5, linestyle='-')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, pct_changes)):
            width = bar.get_width()
            label_x = width + (0.5 if width > 0 else -0.5)
            ax.text(label_x, bar.get_y() + bar.get_height() / 2,
                    f'{val:+.1f}%', ha='left' if width > 0 else 'right',
                    va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def plot_trajectory_comparison(results: Dict[str, Dict], output_path: Path, max_samples: int = 3):
    """
    Plot 3D trajectory comparison for different ablation configurations.
    """
    configs = ["full_model", "wo_sensor", "wo_robot_state", "wo_both"]
    config_labels = {
        "full_model": "Full Model (Baseline)",
        "wo_sensor": "w/o Sensor",
        "wo_robot_state": "w/o Robot State",
        "wo_both": "w/o Both (VL only)",
    }

    # Get GT from baseline
    gt_traj, _ = extract_sample_trajectories(results["full_model"], max_samples)

    fig = plt.figure(figsize=(15, 5 * max_samples))

    for sample_idx in range(min(max_samples, gt_traj.shape[0])):
        ax = fig.add_subplot(max_samples, 1, sample_idx + 1, projection='3d')

        # Plot GT trajectory (cumulative sum of deltas)
        gt_xyz = gt_traj[sample_idx, :, POSITION_DIMS]  # (H, 3)
        gt_cumsum = np.cumsum(gt_xyz, axis=0) * 1000  # to mm

        ax.plot(gt_cumsum[:, 0], gt_cumsum[:, 1], gt_cumsum[:, 2],
                'k-o', label='Ground Truth', linewidth=3, markersize=6, alpha=0.9)

        # Plot predicted trajectories for each config
        for config_name in configs:
            if config_name not in results:
                continue

            _, pred_traj = extract_sample_trajectories(results[config_name], max_samples)

            if sample_idx >= pred_traj.shape[0]:
                continue

            pred_xyz = pred_traj[sample_idx, :, POSITION_DIMS]  # (H, 3)
            pred_cumsum = np.cumsum(pred_xyz, axis=0) * 1000  # to mm

            ax.plot(pred_cumsum[:, 0], pred_cumsum[:, 1], pred_cumsum[:, 2],
                    '--s', label=config_labels[config_name],
                    color=COLORS[config_name],
                    linewidth=2, markersize=5, alpha=0.8)

        ax.set_xlabel('X (mm)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Y (mm)', fontsize=10, fontweight='bold')
        ax.set_zlabel('Z (mm)', fontsize=10, fontweight='bold')
        ax.set_title(f'3D Trajectory Sample {sample_idx + 1}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def generate_summary_table(results: Dict[str, Dict], output_path: Path):
    """Generate a summary table with all key metrics."""
    configs = list(results.keys())

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    # Prepare table data
    headers = [
        "Configuration",
        "Pos RMSE\n(mm)",
        "Pos MAE\n(mm)",
        "Rot RMSE\n(deg)",
        "Rot MAE\n(deg)",
        "Success\nRate (%)",
        "Gripper\nAcc (%)",
    ]

    table_data = []
    for config in configs:
        metrics = results[config]["overall_metrics"]
        row = [
            CONFIG_LABELS[config],
            f"{metrics['position']['rmse_mm']:.3f}",
            f"{metrics['position']['mae_mm']:.3f}",
            f"{metrics['rotation']['rmse_deg']:.3f}",
            f"{metrics['rotation']['mae_deg']:.3f}",
            f"{metrics['position']['success_rate'] * 100:.2f}",
            f"{(metrics['gripper']['accuracy'] or 0.0) * 100:.2f}",
        ]
        table_data.append(row)

    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colWidths=[0.20, 0.13, 0.13, 0.13, 0.13, 0.14, 0.14],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#4A90E2')
        cell.set_text_props(weight='bold', color='white')

    # Style rows with config colors
    for i, config in enumerate(configs):
        cell = table[(i + 1, 0)]
        cell.set_facecolor(COLORS[config])
        cell.set_text_props(weight='bold', color='white')

        # Alternate row colors for readability
        bg_color = '#F5F5F5' if i % 2 == 0 else '#FFFFFF'
        for j in range(1, len(headers)):
            table[(i + 1, j)].set_facecolor(bg_color)

    # Highlight best values in each column (excluding config name)
    for col_idx in range(1, len(headers)):
        values = [float(table_data[i][col_idx]) for i in range(len(configs))]

        # For success rate and gripper acc, higher is better; for errors, lower is better
        if col_idx >= 5:  # Success Rate and Gripper Acc
            best_idx = np.argmax(values)
        else:  # Error metrics
            best_idx = np.argmin(values)

        cell = table[(best_idx + 1, col_idx)]
        cell.set_text_props(weight='bold', color='green')

    plt.title("Ablation Study: Summary Table", fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ablation study comparison visualizations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--full-model", type=str, required=True,
                        help="Path to full model evaluation results JSON")
    parser.add_argument("--wo-sensor", type=str, required=True,
                        help="Path to w/o sensor evaluation results JSON")
    parser.add_argument("--wo-robot-state", type=str, required=True,
                        help="Path to w/o robot state evaluation results JSON")
    parser.add_argument("--wo-both", type=str, required=True,
                        help="Path to w/o both evaluation results JSON")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save comparison plots")
    parser.add_argument("--max-trajectory-samples", type=int, default=3,
                        help="Number of sample trajectories to visualize")

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🔍 Loading evaluation results...")

    # Load all results
    results = {
        "full_model": load_results(Path(args.full_model)),
        "wo_sensor": load_results(Path(args.wo_sensor)),
        "wo_robot_state": load_results(Path(args.wo_robot_state)),
        "wo_both": load_results(Path(args.wo_both)),
    }

    print(f"✅ Loaded {len(results)} configurations")

    # Generate visualizations
    print("\n📊 Generating comparison visualizations...")

    plot_overall_metrics_comparison(
        results,
        output_dir / "ablation_overall_metrics_comparison.png"
    )

    plot_per_dimension_comparison(
        results,
        output_dir / "ablation_per_dimension_comparison.png"
    )

    plot_relative_performance(
        results,
        output_dir / "ablation_relative_performance.png"
    )

    plot_trajectory_comparison(
        results,
        output_dir / "ablation_trajectory_comparison.png",
        max_samples=args.max_trajectory_samples
    )

    generate_summary_table(
        results,
        output_dir / "ablation_summary_table.png"
    )

    # Print summary to console
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)

    for config in ["full_model", "wo_sensor", "wo_robot_state", "wo_both"]:
        metrics = results[config]["overall_metrics"]
        print(f"\n{CONFIG_LABELS[config]}:")
        print(f"  Position RMSE:   {metrics['position']['rmse_mm']:.3f} mm")
        print(f"  Rotation RMSE:   {metrics['rotation']['rmse_deg']:.3f} deg")
        print(f"  Success Rate:    {metrics['position']['success_rate'] * 100:.2f}%")
        print(f"  Gripper Acc:     {(metrics['gripper']['accuracy'] or 0.0) * 100:.2f}%")

    print("\n" + "=" * 80)
    print("\n✅ All comparison plots saved to:", output_dir)
    print("\nGenerated files:")
    print("  - ablation_overall_metrics_comparison.png")
    print("  - ablation_per_dimension_comparison.png")
    print("  - ablation_relative_performance.png")
    print("  - ablation_trajectory_comparison.png")
    print("  - ablation_summary_table.png")
    print("")


if __name__ == "__main__":
    main()
