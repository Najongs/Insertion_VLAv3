#!/usr/bin/env python3
"""
시각화 유틸: 비동기 실시간 추론 결과(JSON)를 불러와서
각 delta action 축과 추론 지연을 한 번에 확인할 수 있도록 그래프로 저장합니다.

python visualize_inference_results.py --input Real_Inference/async_inference_20251117_032356/inference_results_20251117_032356.json \
    --output Real_Inference/async_inference_20251117_032356/inference_plot.png

python visualize_inference_results.py --input Real_Inference/async_inference_20251117_032356/inference_results_20251117_032356.json \
    --output Real_Inference/async_inference_20251117_032356/inference_plot_v2.png \
    --traj-output Real_Inference/async_inference_20251117_032356/

"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


ACTION_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


def load_inference_results(path: Path) -> List[dict]:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON 형식이 리스트가 아닙니다.")
    if not data:
        raise ValueError("JSON에 저장된 결과가 없습니다.")
    return data


def flatten_actions(
    records: List[dict], step_interval: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    Returns:
        times: 전체 action 시각 (첫 추론 기준 상대값, 초)
        actions: (N, 7) array
        inference_idx: 각 action이 어느 추론에서 나왔는지 (0 시작)
        step_markers: 각 추론이 시작되는 인덱스 리스트
    """
    first_ts = records[0].get("timestamp", 0.0)
    action_list = []
    time_list = []
    inference_indices = []
    inference_markers = [0]

    for inf_idx, record in enumerate(records):
        actions = np.asarray(record.get("actions"), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != len(ACTION_NAMES):
            raise ValueError(
                f"{inf_idx}번째 항목의 action shape가 예상과 다릅니다: {actions.shape}"
            )

        ts = float(record.get("timestamp", first_ts))
        base_time = ts - first_ts

        for step_idx in range(actions.shape[0]):
            action_list.append(actions[step_idx])
            time_list.append(base_time + step_idx * step_interval)
            inference_indices.append(inf_idx)

        inference_markers.append(len(action_list))

    actions_arr = np.stack(action_list, axis=0)
    times_arr = np.asarray(time_list, dtype=np.float32)
    inference_indices_arr = np.asarray(inference_indices, dtype=np.int32)

    return times_arr, actions_arr, inference_indices_arr, inference_markers


def plot_results(
    times: np.ndarray,
    actions: np.ndarray,
    inference_indices: np.ndarray,
    inference_markers: List[int],
    inference_records: List[dict],
    output_path: Path,
    trajectory_output_path: Path,
) -> None:
    num_plots = len(ACTION_NAMES) + 1  # 마지막은 inference latency
    rows = 4
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0, vmax=max(1, inference_indices.max()))

    for i, name in enumerate(ACTION_NAMES):
        ax = axes[i]
        ax.scatter(
            times,
            actions[:, i],
            c=inference_indices,
            cmap=cmap,
            norm=norm,
            s=10,
            alpha=0.8,
            edgecolors="none",
        )
        ax.set_ylabel(name)
        ax.grid(True, linestyle="--", alpha=0.3)

    # Inference latency subplot
    latency_ax = axes[-1]
    inf_times = [rec.get("timestamp", 0.0) - inference_records[0].get("timestamp", 0.0) for rec in inference_records]
    inf_latency = [rec.get("inference_time", 0.0) * 1000.0 for rec in inference_records]
    latency_ax.plot(inf_times, inf_latency, marker="o")
    latency_ax.set_ylabel("Latency (ms)")
    latency_ax.set_xlabel("Relative Time (s)")
    latency_ax.grid(True, linestyle="--", alpha=0.3)

    # Set global x-axis label
    axes[-2].set_xlabel("Relative Time (s)")

    # Vertical markers for inference boundaries
    for ax in axes[:-1]:
        for idx in inference_markers:
            if idx <= 0 or idx >= times.shape[0]:
                continue
            ax.axvline(times[idx], color="gray", linestyle=":", linewidth=0.6, alpha=0.6)

    # cbar = fig.colorbar(
    #     plt.cm.ScalarMappable(cmap=cmap, norm=norm),
    #     ax=axes[:-1],
    #     orientation="vertical",
    #     shrink=0.6,
    #     label="Inference Index",
    # )
    # cbar.ax.tick_params(labelsize=8)

    fig.suptitle("Async Inference Action Timeline", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    total_xyz = actions[:, :3].sum(axis=0)
    fig.text(
        0.02,
        0.98,
        f"Σdx={total_xyz[0]:+.4f}, Σdy={total_xyz[1]:+.4f}, Σdz={total_xyz[2]:+.4f}",
        fontsize=10,
        va="top",
    )

    fig.savefig(output_path, dpi=200)
    print(f"✅ 시각화 저장 완료: {output_path}")

    # 3D trajectory plot
    traj = np.cumsum(actions[:, :3], axis=0)
    fig3d = plt.figure(figsize=(8, 6))
    ax3d = fig3d.add_subplot(111, projection="3d")
    ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2], marker="o", linewidth=1.5)
    ax3d.set_xlabel("Σdx")
    ax3d.set_ylabel("Σdy")
    ax3d.set_zlabel("Σdz")
    ax3d.set_title(
        "Cumulative XYZ trajectory\n"
        f"Total Δ = ({total_xyz[0]:+.4f}, {total_xyz[1]:+.4f}, {total_xyz[2]:+.4f})"
    )
    ax3d.scatter([traj[0, 0]], [traj[0, 1]], [traj[0, 2]], c="green", label="start")
    ax3d.scatter([traj[-1, 0]], [traj[-1, 1]], [traj[-1, 2]], c="red", label="end")
    ax3d.legend()
    # ax3d.grid(True, linestyle="--", alpha=0.3)
    fig3d.tight_layout()
    fig3d.savefig(trajectory_output_path, dpi=200)
    print(f"✅ 3D 누적 이동 경로 저장: {trajectory_output_path}")


def main():
    parser = argparse.ArgumentParser(description="Async inference JSON 시각화")
    parser.add_argument("--input", required=True, type=Path, help="inference_results_*.json 경로")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="결과 PNG 저장 경로 (기본: JSON 파일명+_plot.png)",
    )
    parser.add_argument(
        "--traj-output",
        type=Path,
        default=None,
        help="3D 궤적 PNG 경로 (기본: JSON 파일명+_traj.png)",
    )
    parser.add_argument(
        "--step-interval",
        type=float,
        default=0.1,
        help="action 간격(초) - 기본 0.1s",
    )
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    if args.output:
        output_path = args.output.expanduser().resolve()
    else:
        output_path = input_path.with_suffix("").with_name(input_path.stem + "_plot.png")
    if args.traj_output:
        traj_output_path = args.traj_output.expanduser().resolve()
    else:
        traj_output_path = input_path.with_suffix("").with_name(input_path.stem + "_traj.png")

    records = load_inference_results(input_path)
    times, actions, inference_indices, inference_markers = flatten_actions(
        records, args.step_interval
    )
    plot_results(
        times,
        actions,
        inference_indices,
        inference_markers,
        records,
        output_path,
        traj_output_path,
    )


if __name__ == "__main__":
    main()
