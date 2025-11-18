#!/usr/bin/env python3
"""
View Ablation Study Comparison Visualization Script

Compares evaluation results from different view configurations:
- All views (baseline)
- Different number of views (4, 3, 2, 1)
- Each individual view

Generates:
1. Performance vs Number of Views (line plots)
2. Single View Comparison (bar charts)
3. Inference Time vs Performance Trade-off
4. 3D Trajectory Comparison (GT vs Predicted for different view configs)
5. Summary Table with all metrics and timing

Usage:
    python evaluation_results/plot_view_ablation_comparison.py \
        --all-views evaluation_results/view_ablation_study/all_views/evaluation_results_*.json \
        --views-4 evaluation_results/view_ablation_study/views_0_1_2_3/evaluation_results_*.json \
        --views-3 evaluation_results/view_ablation_study/views_0_1_2/evaluation_results_*.json \
        --views-2 evaluation_results/view_ablation_study/views_0_1/evaluation_results_*.json \
        --view-0 evaluation_results/view_ablation_study/view_0/evaluation_results_*.json \
        --view-1 evaluation_results/view_ablation_study/view_1/evaluation_results_*.json \
        --view-2 evaluation_results/view_ablation_study/view_2/evaluation_results_*.json \
        --view-3 evaluation_results/view_ablation_study/view_3/evaluation_results_*.json \
        --view-4 evaluation_results/view_ablation_study/view_4/evaluation_results_*.json \
        --output-dir evaluation_results/view_ablation_study/comparison_plots \
        --max-trajectory-samples 3
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


ACTION_LABELS = ["dx", "dy", "dz", "rx", "ry", "rz", "gripper"]
POSITION_DIMS = [0, 1, 2]  # x, y, z


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


def plot_performance_vs_num_views(results_by_num_views: Dict[int, Dict], output_path: Path):
    """
    Plot performance metrics vs number of views.

    Args:
        results_by_num_views: Dict mapping num_views -> results dict
    """
    num_views_list = sorted(results_by_num_views.keys())

    # Extract metrics
    pos_rmse = [results_by_num_views[n]["overall_metrics"]["position"]["rmse_mm"] for n in num_views_list]
    rot_rmse = [results_by_num_views[n]["overall_metrics"]["rotation"]["rmse_deg"] for n in num_views_list]
    success_rate = [results_by_num_views[n]["overall_metrics"]["position"]["success_rate"] * 100 for n in num_views_list]
    inference_time = [results_by_num_views[n]["overall_timing"]["avg_time_per_sample_ms"] for n in num_views_list]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("View Ablation Study: Performance vs Number of Views", fontsize=16, fontweight="bold")

    # Position RMSE
    ax = axes[0, 0]
    ax.plot(num_views_list, pos_rmse, 'o-', linewidth=2, markersize=8, color='#E74C3C')
    ax.set_xlabel("Number of Views", fontsize=11, fontweight="bold")
    ax.set_ylabel("Position RMSE (mm)", fontsize=11, fontweight="bold")
    ax.set_title("Position Accuracy", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xticks(num_views_list)

    # Annotate values
    for x, y in zip(num_views_list, pos_rmse):
        ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=9, fontweight='bold')

    # Rotation RMSE
    ax = axes[0, 1]
    ax.plot(num_views_list, rot_rmse, 'o-', linewidth=2, markersize=8, color='#3498DB')
    ax.set_xlabel("Number of Views", fontsize=11, fontweight="bold")
    ax.set_ylabel("Rotation RMSE (deg)", fontsize=11, fontweight="bold")
    ax.set_title("Rotation Accuracy", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xticks(num_views_list)

    for x, y in zip(num_views_list, rot_rmse):
        ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=9, fontweight='bold')

    # Success Rate
    ax = axes[1, 0]
    ax.plot(num_views_list, success_rate, 'o-', linewidth=2, markersize=8, color='#2ECC71')
    ax.set_xlabel("Number of Views", fontsize=11, fontweight="bold")
    ax.set_ylabel("Success Rate (%)", fontsize=11, fontweight="bold")
    ax.set_title("Success Rate", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xticks(num_views_list)
    ax.set_ylim([0, 100])

    for x, y in zip(num_views_list, success_rate):
        ax.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=9, fontweight='bold')

    # Inference Time
    ax = axes[1, 1]
    ax.plot(num_views_list, inference_time, 'o-', linewidth=2, markersize=8, color='#9B59B6')
    ax.set_xlabel("Number of Views", fontsize=11, fontweight="bold")
    ax.set_ylabel("Inference Time (ms/sample)", fontsize=11, fontweight="bold")
    ax.set_title("Computational Cost", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xticks(num_views_list)

    for x, y in zip(num_views_list, inference_time):
        ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def plot_single_view_comparison(single_view_results: Dict[int, Dict], output_path: Path):
    """
    Compare performance across different single views.

    Args:
        single_view_results: Dict mapping view_index -> results dict
    """
    view_indices = sorted(single_view_results.keys())
    view_labels = [f"View {i}" for i in view_indices]

    # Extract metrics
    pos_rmse = [single_view_results[v]["overall_metrics"]["position"]["rmse_mm"] for v in view_indices]
    rot_rmse = [single_view_results[v]["overall_metrics"]["rotation"]["rmse_deg"] for v in view_indices]
    success_rate = [single_view_results[v]["overall_metrics"]["position"]["success_rate"] * 100 for v in view_indices]
    inference_time = [single_view_results[v]["overall_timing"]["avg_time_per_sample_ms"] for v in view_indices]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("View Ablation Study: Single View Comparison", fontsize=16, fontweight="bold")

    x = np.arange(len(view_indices))
    width = 0.6
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

    # Position RMSE
    ax = axes[0, 0]
    bars = ax.bar(x, pos_rmse, width, color=colors[:len(view_indices)], alpha=0.8, edgecolor='black')
    ax.set_ylabel("Position RMSE (mm)", fontsize=11, fontweight="bold")
    ax.set_title("Position Accuracy by View", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(view_labels)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    for bar, val in zip(bars, pos_rmse):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Rotation RMSE
    ax = axes[0, 1]
    bars = ax.bar(x, rot_rmse, width, color=colors[:len(view_indices)], alpha=0.8, edgecolor='black')
    ax.set_ylabel("Rotation RMSE (deg)", fontsize=11, fontweight="bold")
    ax.set_title("Rotation Accuracy by View", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(view_labels)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    for bar, val in zip(bars, rot_rmse):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Success Rate
    ax = axes[1, 0]
    bars = ax.bar(x, success_rate, width, color=colors[:len(view_indices)], alpha=0.8, edgecolor='black')
    ax.set_ylabel("Success Rate (%)", fontsize=11, fontweight="bold")
    ax.set_title("Success Rate by View", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(view_labels)
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    for bar, val in zip(bars, success_rate):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Inference Time
    ax = axes[1, 1]
    bars = ax.bar(x, inference_time, width, color=colors[:len(view_indices)], alpha=0.8, edgecolor='black')
    ax.set_ylabel("Inference Time (ms/sample)", fontsize=11, fontweight="bold")
    ax.set_title("Inference Time by View", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(view_labels)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    for bar, val in zip(bars, inference_time):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def plot_performance_inference_tradeoff(all_results: Dict[str, Dict], output_path: Path):
    """
    Plot performance vs inference time trade-off (Pareto frontier analysis).
    """
    config_names = []
    pos_rmse = []
    inference_time = []
    success_rate = []

    for name, results in all_results.items():
        config_names.append(name)
        pos_rmse.append(results["overall_metrics"]["position"]["rmse_mm"])
        inference_time.append(results["overall_timing"]["avg_time_per_sample_ms"])
        success_rate.append(results["overall_metrics"]["position"]["success_rate"] * 100)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("View Ablation Study: Performance-Inference Time Trade-off", fontsize=14, fontweight="bold")

    # Position RMSE vs Inference Time
    ax = axes[0]
    scatter = ax.scatter(inference_time, pos_rmse, s=150, c=success_rate, cmap='RdYlGn',
                         edgecolors='black', linewidths=2, alpha=0.8)
    ax.set_xlabel("Inference Time (ms/sample)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Position RMSE (mm)", fontsize=11, fontweight="bold")
    ax.set_title("Lower is Better (Faster & More Accurate)", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, linestyle='--')

    # Annotate points
    for i, name in enumerate(config_names):
        ax.annotate(name, (inference_time[i], pos_rmse[i]),
                    textcoords="offset points", xytext=(0, 8),
                    ha='center', fontsize=8, fontweight='bold')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Success Rate (%)', fontsize=10, fontweight='bold')

    # Success Rate vs Inference Time
    ax = axes[1]
    scatter = ax.scatter(inference_time, success_rate, s=150, c=pos_rmse, cmap='RdYlGn_r',
                         edgecolors='black', linewidths=2, alpha=0.8)
    ax.set_xlabel("Inference Time (ms/sample)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Success Rate (%)", fontsize=11, fontweight="bold")
    ax.set_title("Higher Success with Lower Time is Better", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim([0, 100])

    # Annotate points
    for i, name in enumerate(config_names):
        ax.annotate(name, (inference_time[i], success_rate[i]),
                    textcoords="offset points", xytext=(0, 8),
                    ha='center', fontsize=8, fontweight='bold')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Position RMSE (mm)', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def plot_trajectory_comparison(all_results: Dict[str, Dict], output_path: Path, max_samples: int = 3):
    """
    Plot 3D trajectory comparison for different view configurations.
    """
    configs_to_plot = ["all_views", "views_0_1", "view_0"]
    config_labels = {
        "all_views": "All Views (Baseline)",
        "views_0_1": "2 Views (0,1)",
        "view_0": "Single View 0",
    }
    colors_map = {
        "all_views": "#2ECC71",
        "views_0_1": "#3498DB",
        "view_0": "#E74C3C",
    }

    # Get GT from baseline
    gt_traj, _ = extract_sample_trajectories(all_results["all_views"], max_samples)

    fig = plt.figure(figsize=(15, 5 * max_samples))

    for sample_idx in range(min(max_samples, gt_traj.shape[0])):
        ax = fig.add_subplot(max_samples, 1, sample_idx + 1, projection='3d')

        # Plot GT trajectory (cumulative sum of deltas)
        gt_xyz = gt_traj[sample_idx, :, POSITION_DIMS]  # (H, 3)
        gt_cumsum = np.cumsum(gt_xyz, axis=0) * 1000  # to mm

        ax.plot(gt_cumsum[:, 0], gt_cumsum[:, 1], gt_cumsum[:, 2],
                'k-o', label='Ground Truth', linewidth=3, markersize=6, alpha=0.9)

        # Plot predicted trajectories for each config
        for config_name in configs_to_plot:
            if config_name not in all_results:
                continue

            _, pred_traj = extract_sample_trajectories(all_results[config_name], max_samples)

            if sample_idx >= pred_traj.shape[0]:
                continue

            pred_xyz = pred_traj[sample_idx, :, POSITION_DIMS]  # (H, 3)
            pred_cumsum = np.cumsum(pred_xyz, axis=0) * 1000  # to mm

            ax.plot(pred_cumsum[:, 0], pred_cumsum[:, 1], pred_cumsum[:, 2],
                    '--s', label=config_labels[config_name],
                    color=colors_map[config_name],
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


def generate_summary_table(all_results: Dict[str, Dict], output_path: Path):
    """Generate a summary table with all configurations."""
    config_order = [
        "all_views", "views_0_1_2_3", "views_0_1_2", "views_0_1",
        "view_0", "view_1", "view_2", "view_3", "view_4"
    ]

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')

    # Prepare table data
    headers = [
        "Configuration",
        "Pos RMSE\n(mm)",
        "Rot RMSE\n(deg)",
        "Success\nRate (%)",
        "Gripper\nAcc (%)",
        "Inference\nTime (ms)",
    ]

    table_data = []
    for config_name in config_order:
        if config_name not in all_results:
            continue

        results = all_results[config_name]
        metrics = results["overall_metrics"]
        timing = results["overall_timing"]

        row = [
            config_name.replace('_', ' ').title(),
            f"{metrics['position']['rmse_mm']:.3f}",
            f"{metrics['rotation']['rmse_deg']:.3f}",
            f"{metrics['position']['success_rate'] * 100:.2f}",
            f"{(metrics['gripper']['accuracy'] or 0.0) * 100:.2f}",
            f"{timing['avg_time_per_sample_ms']:.2f}",
        ]
        table_data.append(row)

    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#4A90E2')
        cell.set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(len(table_data)):
        bg_color = '#F5F5F5' if i % 2 == 0 else '#FFFFFF'
        for j in range(len(headers)):
            table[(i + 1, j)].set_facecolor(bg_color)

    # Highlight best values in each column (excluding config name)
    for col_idx in range(1, len(headers)):
        values = [float(table_data[i][col_idx]) for i in range(len(table_data))]

        # For success rate, higher is better; for others, lower is better
        if col_idx in [3, 4]:  # Success Rate and Gripper Acc
            best_idx = np.argmax(values)
        else:  # Error metrics and time
            best_idx = np.argmin(values)

        cell = table[(best_idx + 1, col_idx)]
        cell.set_text_props(weight='bold', color='green')

    plt.title("View Ablation Study: Summary Table", fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate view ablation study comparison visualizations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--all-views", type=str, required=True)
    parser.add_argument("--views-4", type=str, required=True)
    parser.add_argument("--views-3", type=str, required=True)
    parser.add_argument("--views-2", type=str, required=True)
    parser.add_argument("--view-0", type=str, required=True)
    parser.add_argument("--view-1", type=str, required=True)
    parser.add_argument("--view-2", type=str, required=True)
    parser.add_argument("--view-3", type=str, required=True)
    parser.add_argument("--view-4", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-trajectory-samples", type=int, default=3)

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🔍 Loading evaluation results...")

    # Load all results
    all_results = {
        "all_views": load_results(Path(args.all_views)),
        "views_0_1_2_3": load_results(Path(args.views_4)),
        "views_0_1_2": load_results(Path(args.views_3)),
        "views_0_1": load_results(Path(args.views_2)),
        "view_0": load_results(Path(args.view_0)),
        "view_1": load_results(Path(args.view_1)),
        "view_2": load_results(Path(args.view_2)),
        "view_3": load_results(Path(args.view_3)),
        "view_4": load_results(Path(args.view_4)),
    }

    print(f"✅ Loaded {len(all_results)} configurations")

    # Organize results by type
    results_by_num_views = {
        5: all_results["all_views"],
        4: all_results["views_0_1_2_3"],
        3: all_results["views_0_1_2"],
        2: all_results["views_0_1"],
        1: all_results["view_0"],  # Use view_0 as representative single view
    }

    single_view_results = {
        0: all_results["view_0"],
        1: all_results["view_1"],
        2: all_results["view_2"],
        3: all_results["view_3"],
        4: all_results["view_4"],
    }

    # Generate visualizations
    print("\n📊 Generating comparison visualizations...")

    plot_performance_vs_num_views(
        results_by_num_views,
        output_dir / "view_ablation_performance_vs_num_views.png"
    )

    plot_single_view_comparison(
        single_view_results,
        output_dir / "view_ablation_single_view_comparison.png"
    )

    plot_performance_inference_tradeoff(
        all_results,
        output_dir / "view_ablation_performance_inference_tradeoff.png"
    )

    plot_trajectory_comparison(
        all_results,
        output_dir / "view_ablation_trajectory_comparison.png",
        max_samples=args.max_trajectory_samples
    )

    generate_summary_table(
        all_results,
        output_dir / "view_ablation_summary_table.png"
    )

    # Print summary
    print("\n" + "=" * 80)
    print("VIEW ABLATION STUDY SUMMARY")
    print("=" * 80)

    for name in ["all_views", "views_0_1_2_3", "views_0_1_2", "views_0_1", "view_0"]:
        metrics = all_results[name]["overall_metrics"]
        timing = all_results[name]["overall_timing"]
        print(f"\n{name.replace('_', ' ').title()}:")
        print(f"  Position RMSE:     {metrics['position']['rmse_mm']:.3f} mm")
        print(f"  Success Rate:      {metrics['position']['success_rate'] * 100:.2f}%")
        print(f"  Inference Time:    {timing['avg_time_per_sample_ms']:.2f} ms/sample")

    print("\n" + "=" * 80)
    print("\n✅ All comparison plots saved to:", output_dir)
    print("\nGenerated files:")
    print("  - view_ablation_performance_vs_num_views.png")
    print("  - view_ablation_single_view_comparison.png")
    print("  - view_ablation_performance_inference_tradeoff.png")
    print("  - view_ablation_trajectory_comparison.png")
    print("  - view_ablation_summary_table.png")
    print("")


if __name__ == "__main__":
    main()
