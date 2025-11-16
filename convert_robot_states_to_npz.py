#!/usr/bin/env python3
"""
Convert robot_states.csv to robot_states.npz for faster loading

Usage:
    python convert_robot_states_to_npz.py <dataset_dir>
    python convert_robot_states_to_npz.py /home/najo/NAS/VLA/dataset/New_dataset5/Eye_trocar  # Convert all episodes
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def convert_csv_to_npz(csv_path: Path, output_path: Path = None):
    """
    Convert robot_states.csv to robot_states.npz

    Args:
        csv_path: Path to robot_states.csv
        output_path: Output path for .npz file (default: same dir as csv)
    """
    if output_path is None:
        output_path = csv_path.parent / "robot_states.npz"

    # Read CSV with only required columns
    joint_cols = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
    pose_cols = ['pose_x', 'pose_y', 'pose_z', 'pose_a', 'pose_b', 'pose_r']
    timestamp_col = ['recv_timestamp']

    use_cols = timestamp_col + joint_cols + pose_cols

    try:
        df = pd.read_csv(csv_path, usecols=use_cols)
    except Exception as e:
        print(f"⚠️  Failed to read {csv_path}: {e}")
        return False

    # Extract data
    timestamps = df['recv_timestamp'].to_numpy(dtype=np.float64)
    joints = df[joint_cols].to_numpy(dtype=np.float32)
    poses = df[pose_cols].to_numpy(dtype=np.float32)

    # Combine joint + pose (12 dims)
    robot_states = np.concatenate([joints, poses], axis=1)  # (N, 12)

    # Save to NPZ with compression
    np.savez_compressed(
        output_path,
        timestamps=timestamps,
        joints=joints,
        poses=poses,
        robot_states=robot_states  # Combined data
    )

    # Print stats
    csv_size = csv_path.stat().st_size / (1024 * 1024)  # MB
    npz_size = output_path.stat().st_size / (1024 * 1024)  # MB
    compression_ratio = (1 - npz_size / csv_size) * 100

    print(f"✅ {csv_path.parent.name}")
    print(f"   CSV: {csv_size:.2f} MB → NPZ: {npz_size:.2f} MB ({compression_ratio:.1f}% reduction)")
    print(f"   Samples: {len(robot_states)}, Shape: {robot_states.shape}")

    return True


def convert_all_episodes(dataset_root: Path):
    """
    Convert all episodes in a dataset directory

    Args:
        dataset_root: Root directory containing episode folders
    """
    if not dataset_root.exists():
        print(f"❌ Directory not found: {dataset_root}")
        return

    # Find all robot_states.csv or robot_state_*.csv files
    csv_files = list(dataset_root.rglob("robot_states.csv"))
    if not csv_files:
        csv_files = list(dataset_root.rglob("robot_state_*.csv"))


    if not csv_files:
        print(f"⚠️  No robot_states.csv or robot_state_*.csv files found in {dataset_root}")
        return

    print(f"🔍 Found {len(csv_files)} robot state csv files")
    print(f"📁 Converting in: {dataset_root}\n")

    success_count = 0
    failed_files = []

    for csv_path in tqdm(csv_files, desc="Converting"):
        if convert_csv_to_npz(csv_path):
            success_count += 1
        else:
            failed_files.append(csv_path)

    print(f"\n✅ Conversion complete!")
    print(f"   Success: {success_count}/{len(csv_files)}")

    if failed_files:
        print(f"   Failed: {len(failed_files)}")
        for f in failed_files:
            print(f"     - {f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_robot_states_to_npz.py <dataset_dir>")
        print("Example: python convert_robot_states_to_npz.py /path/to/New_dataset/Blue_point")
        sys.exit(1)

    dataset_path = Path(sys.argv[1])

    if dataset_path.is_file() and dataset_path.name == "robot_states.csv":
        # Single file conversion
        convert_csv_to_npz(dataset_path)
    elif dataset_path.is_dir():
        # Batch conversion
        convert_all_episodes(dataset_path)
    else:
        print(f"❌ Invalid path: {dataset_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
