"""
Trajectory Visualization from Delta Actions

Reconstructs and visualizes the complete trajectory from delta actions
loaded by new_format_dataset.py. Shows both 3D position path and orientation changes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from pathlib import Path
import sys

# Add parent directory to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from vla_datasets.unified_dataset import UnifiedVLADataset


def reconstruct_trajectory_from_deltas(initial_pose, delta_actions):
    """
    Reconstruct full trajectory from initial pose and delta actions.

    Args:
        initial_pose: [x, y, z, rx, ry, rz] initial pose in degrees
        delta_actions: [N, 7] array of delta actions [dx, dy, dz, rot_vec_x, rot_vec_y, rot_vec_z, gripper]

    Returns:
        positions: [N+1, 3] array of positions
        orientations: [N+1, 3] array of euler angles in degrees
        gripper_states: [N] array of gripper states
    """
    num_steps = len(delta_actions)
    positions = np.zeros((num_steps + 1, 3))
    orientations = np.zeros((num_steps + 1, 3))
    gripper_states = delta_actions[:, 6]

    # Set initial state
    positions[0] = initial_pose[:3]
    orientations[0] = initial_pose[3:]

    # Current rotation
    current_rotation = Rotation.from_euler('xyz', initial_pose[3:], degrees=True)

    # Apply each delta action
    for i in range(num_steps):
        delta_action = delta_actions[i]

        # Translation delta
        delta_trans = delta_action[:3]
        positions[i + 1] = positions[i] + delta_trans

        # Rotation delta (rotation vector)
        delta_rot_vec = delta_action[3:6]
        delta_rotation = Rotation.from_rotvec(delta_rot_vec)

        # Apply rotation
        current_rotation = delta_rotation * current_rotation
        orientations[i + 1] = current_rotation.as_euler('xyz', degrees=True)

    return positions, orientations, gripper_states


def plot_trajectory_3d(positions, orientations, gripper_states, title="Robot Trajectory"):
    """
    Plot 3D trajectory with orientation arrows and gripper state colors.

    Args:
        positions: [N, 3] positions
        orientations: [N, 3] euler angles in degrees
        gripper_states: [N-1] gripper states
        title: plot title
    """
    fig = plt.figure(figsize=(15, 5))

    # 3D trajectory plot
    ax1 = fig.add_subplot(131, projection='3d')

    # Color by gripper state
    colors = plt.cm.RdYlGn(gripper_states)

    # Plot trajectory line
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2],
             'b-', alpha=0.3, linewidth=1)

    # Plot points colored by gripper state
    for i in range(len(positions) - 1):
        ax1.scatter(positions[i, 0], positions[i, 1], positions[i, 2],
                   c=[colors[i]], s=20, alpha=0.6)

    # Mark start and end
    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
               c='green', s=200, marker='o', label='Start', edgecolors='black', linewidths=2)
    ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
               c='red', s=200, marker='*', label='End', edgecolors='black', linewidths=2)

    # Plot orientation arrows at key points
    arrow_step = max(1, len(positions) // 10)
    for i in range(0, len(positions), arrow_step):
        rot = Rotation.from_euler('xyz', orientations[i], degrees=True)
        direction = rot.apply([0, 0, 0.01])  # Small arrow in tool direction
        ax1.quiver(positions[i, 0], positions[i, 1], positions[i, 2],
                  direction[0], direction[1], direction[2],
                  color='black', alpha=0.5, arrow_length_ratio=0.3)

    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # XY plane view
    ax2 = fig.add_subplot(132)
    ax2.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.3, linewidth=1)
    for i in range(len(positions) - 1):
        ax2.scatter(positions[i, 0], positions[i, 1], c=[colors[i]], s=20, alpha=0.6)
    ax2.scatter(positions[0, 0], positions[0, 1], c='green', s=200, marker='o',
               edgecolors='black', linewidths=2)
    ax2.scatter(positions[-1, 0], positions[-1, 1], c='red', s=200, marker='*',
               edgecolors='black', linewidths=2)
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title('Top View (XY Plane)')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    # XZ plane view
    ax3 = fig.add_subplot(133)
    ax3.plot(positions[:, 0], positions[:, 2], 'b-', alpha=0.3, linewidth=1)
    for i in range(len(positions) - 1):
        ax3.scatter(positions[i, 0], positions[i, 2], c=[colors[i]], s=20, alpha=0.6)
    ax3.scatter(positions[0, 0], positions[0, 2], c='green', s=200, marker='o',
               edgecolors='black', linewidths=2)
    ax3.scatter(positions[-1, 0], positions[-1, 2], c='red', s=200, marker='*',
               edgecolors='black', linewidths=2)
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Z (mm)')
    ax3.set_title('Side View (XZ Plane)')
    ax3.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def plot_trajectory_details(positions, orientations, gripper_states):
    """
    Plot detailed trajectory information over time.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    time_steps = np.arange(len(positions))

    # Position X over time
    ax = axes[0, 0]
    ax.plot(time_steps, positions[:, 0], 'r-', linewidth=2, alpha=0.7)
    ax.scatter(0, positions[0, 0], c='green', s=100, marker='o', edgecolors='black', linewidths=2, zorder=5)
    ax.scatter(len(positions)-1, positions[-1, 0], c='red', s=100, marker='*', edgecolors='black', linewidths=2, zorder=5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('X Position (mm)')
    ax.set_title('X Position Over Time')
    ax.grid(True, alpha=0.3)

    # Position Y over time
    ax = axes[0, 1]
    ax.plot(time_steps, positions[:, 1], 'g-', linewidth=2, alpha=0.7)
    ax.scatter(0, positions[0, 1], c='green', s=100, marker='o', edgecolors='black', linewidths=2, zorder=5)
    ax.scatter(len(positions)-1, positions[-1, 1], c='red', s=100, marker='*', edgecolors='black', linewidths=2, zorder=5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Y Position (mm)')
    ax.set_title('Y Position Over Time')
    ax.grid(True, alpha=0.3)

    # Position Z over time
    ax = axes[0, 2]
    ax.plot(time_steps, positions[:, 2], 'b-', linewidth=2, alpha=0.7)
    ax.scatter(0, positions[0, 2], c='green', s=100, marker='o', edgecolors='black', linewidths=2, zorder=5)
    ax.scatter(len(positions)-1, positions[-1, 2], c='red', s=100, marker='*', edgecolors='black', linewidths=2, zorder=5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Z Position (mm)')
    ax.set_title('Z Position Over Time')
    ax.grid(True, alpha=0.3)

    # Orientation RX over time
    ax = axes[1, 0]
    ax.plot(time_steps, orientations[:, 0], 'r-', linewidth=2, alpha=0.7)
    ax.scatter(0, orientations[0, 0], c='green', s=100, marker='o', edgecolors='black', linewidths=2, zorder=5)
    ax.scatter(len(orientations)-1, orientations[-1, 0], c='red', s=100, marker='*', edgecolors='black', linewidths=2, zorder=5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('RX Orientation (degrees)')
    ax.set_title('RX Orientation Over Time')
    ax.grid(True, alpha=0.3)

    # Orientation RY over time
    ax = axes[1, 1]
    ax.plot(time_steps, orientations[:, 1], 'g-', linewidth=2, alpha=0.7)
    ax.scatter(0, orientations[0, 1], c='green', s=100, marker='o', edgecolors='black', linewidths=2, zorder=5)
    ax.scatter(len(orientations)-1, orientations[-1, 1], c='red', s=100, marker='*', edgecolors='black', linewidths=2, zorder=5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('RY Orientation (degrees)')
    ax.set_title('RY Orientation Over Time')
    ax.grid(True, alpha=0.3)

    # Orientation RZ over time
    ax = axes[1, 2]
    ax.plot(time_steps, orientations[:, 2], 'b-', linewidth=2, alpha=0.7)
    ax.scatter(0, orientations[0, 2], c='green', s=100, marker='o', edgecolors='black', linewidths=2, zorder=5)
    ax.scatter(len(orientations)-1, orientations[-1, 2], c='red', s=100, marker='*', edgecolors='black', linewidths=2, zorder=5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('RZ Orientation (degrees)')
    ax.set_title('RZ Orientation Over Time')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def visualize_episode_trajectory(episode_path, sample_idx=0, save_dir=None, full_episode=False):
    """
    Visualize trajectory for a specific episode.

    Args:
        episode_path: Path to episode directory
        sample_idx: Which sample index to visualize (default 0, ignored if full_episode=True)
        save_dir: Optional directory to save plots
        full_episode: If True, visualize entire episode instead of single sample
    """
    episode_path = Path(episode_path)

    print(f"Loading episode: {episode_path.name}")
    if not full_episode:
        print(f"Sample index: {sample_idx}")
    else:
        print(f"Mode: Full episode")

    # Load dataset
    ds = UnifiedVLADataset(
        data_dir=str(episode_path),
        format='new',
        horizon=8,
        vlm_reuse_count=3,
        sensor_window_size=650,
        robot_window_size=100,
        action_expert_hz=10,
        use_cache=False,
    )

    if len(ds) == 0:
        print("❌ Dataset is empty!")
        return

    print(f"\nDataset info:")
    print(f"  Total samples: {len(ds)}")
    print(f"  Horizon: {ds.horizon}")
    print(f"  Action interval: {ds.action_interval}")
    print(f"  Number of actions: {ds.num_actions}")
    print(f"  Total poses: {len(ds.poses)}")

    if full_episode:
        # Use all poses from the episode
        positions = ds.poses[:, :3]
        orientations = ds.poses[:, 3:]

        # Create dummy gripper states (all open)
        gripper_states = np.ones(len(positions) - 1)

        print(f"\nFull episode trajectory:")
        print(f"  Total steps: {len(positions)}")
        print(f"  Start position: [{positions[0, 0]:.2f}, {positions[0, 1]:.2f}, {positions[0, 2]:.2f}] mm")
        print(f"  End position: [{positions[-1, 0]:.2f}, {positions[-1, 1]:.2f}, {positions[-1, 2]:.2f}] mm")
        print(f"  Total displacement: {np.linalg.norm(positions[-1] - positions[0]):.2f} mm")

        step_sizes = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        print(f"  Max step size: {np.max(step_sizes):.2f} mm")
        print(f"  Min step size: {np.min(step_sizes):.4f} mm")
        print(f"  Mean step size: {np.mean(step_sizes):.4f} mm")

        title_suffix = "Full Episode"
        filename_suffix = "full"
    else:
        # Single sample visualization
        if sample_idx >= len(ds):
            print(f"⚠️ Sample index {sample_idx} out of range. Using 0 instead.")
            sample_idx = 0

        # Get sample
        sample = ds[sample_idx]
        delta_actions = sample['actions'].numpy()

        print(f"\nSample {sample_idx} info:")
        print(f"  Delta actions shape: {delta_actions.shape}")
        print(f"  VLM idx: {sample['vlm_idx']}")
        print(f"  Episode: {sample['episode_id']}")

        # Get initial pose
        action_step_idx = sample_idx * ds.action_interval
        initial_pose = ds.poses[action_step_idx]

        print(f"\nInitial pose at step {action_step_idx}:")
        print(f"  Position: [{initial_pose[0]:.2f}, {initial_pose[1]:.2f}, {initial_pose[2]:.2f}] mm")
        print(f"  Orientation: [{initial_pose[3]:.2f}, {initial_pose[4]:.2f}, {initial_pose[5]:.2f}] deg")

        # Reconstruct trajectory
        positions, orientations, gripper_states = reconstruct_trajectory_from_deltas(
            initial_pose, delta_actions
        )

        print(f"\nReconstructed trajectory:")
        print(f"  Start position: {positions[0]}")
        print(f"  End position: {positions[-1]}")
        print(f"  Total displacement: {np.linalg.norm(positions[-1] - positions[0]):.2f} mm")
        print(f"  Max step size: {np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1)):.2f} mm")
        print(f"  Min step size: {np.min(np.linalg.norm(np.diff(positions, axis=0), axis=1)):.4f} mm")

        title_suffix = f"Sample {sample_idx}"
        filename_suffix = f"sample{sample_idx}"

    # Create visualizations
    task_name = episode_path.parent.name
    title = f"{task_name} - {episode_path.name} ({title_suffix})"

    fig1 = plot_trajectory_3d(positions, orientations, gripper_states, title)
    fig2 = plot_trajectory_details(positions, orientations, gripper_states)

    # Save plots if requested
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        filename_prefix = f"{episode_path.name}_{filename_suffix}"
        fig1.savefig(save_dir / f"{filename_prefix}_3d.png", dpi=150, bbox_inches='tight')
        fig2.savefig(save_dir / f"{filename_prefix}_details.png", dpi=150, bbox_inches='tight')
        print(f"\n✅ Plots saved to {save_dir}/")

    plt.show()


def compare_multiple_episodes(episode_paths, sample_idx=0):
    """
    Compare trajectories from multiple episodes side by side.
    """
    fig = plt.figure(figsize=(5 * len(episode_paths), 5))

    for i, episode_path in enumerate(episode_paths):
        episode_path = Path(episode_path)

        # Load dataset
        ds = UnifiedVLADataset(
            data_dir=str(episode_path),
            format='new',
            horizon=8,
            vlm_reuse_count=3,
            sensor_window_size=650,
            robot_window_size=100,
            action_expert_hz=10,
            use_cache=False,
        )

        if len(ds) == 0:
            continue

        sample = ds[min(sample_idx, len(ds) - 1)]
        delta_actions = sample['actions'].numpy()

        action_step_idx = min(sample_idx, len(ds) - 1) * ds.action_interval
        initial_pose = ds.poses[action_step_idx]

        positions, orientations, gripper_states = reconstruct_trajectory_from_deltas(
            initial_pose, delta_actions
        )

        # Plot
        ax = fig.add_subplot(1, len(episode_paths), i + 1, projection='3d')

        colors = plt.cm.RdYlGn(gripper_states)
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
               'b-', alpha=0.3, linewidth=1)

        for j in range(len(positions) - 1):
            ax.scatter(positions[j, 0], positions[j, 1], positions[j, 2],
                      c=[colors[j]], s=20, alpha=0.6)

        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
                  c='green', s=200, marker='o', edgecolors='black', linewidths=2)
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
                  c='red', s=200, marker='*', edgecolors='black', linewidths=2)

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f"{episode_path.parent.name}\n{episode_path.name}")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Trajectory Comparison (Sample {sample_idx})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize robot trajectory from delta actions")
    parser.add_argument("episode_path", type=str, help="Path to episode directory")
    parser.add_argument("--sample_idx", type=int, default=0, help="Sample index to visualize")
    parser.add_argument("--save_dir", type=str, help="Directory to save plots")
    parser.add_argument("--compare", nargs="+", help="Additional episodes to compare")

    args = parser.parse_args()

    if args.compare:
        # Compare multiple episodes
        all_episodes = [args.episode_path] + args.compare
        compare_multiple_episodes(all_episodes, args.sample_idx)
    else:
        # Visualize single episode
        visualize_episode_trajectory(args.episode_path, args.sample_idx, args.save_dir)
