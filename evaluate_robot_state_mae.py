"""
Evaluate a pre-trained Robot State MAE model.

This script loads a trained MAERobotStateModel and evaluates its reconstruction
performance on a validation set. It handles both 'absolute' and 'delta'
data representations, allowing evaluation either in absolute position space or
in the delta space for direct comparison of reconstruction fidelity.

Evaluation Protocol:
1. Load model and validation dataset based on command-line arguments.
2. For each sample in the validation set:
   a. Get the model's reconstruction of the (normalized) data.
   b. De-normalize both the original data and the reconstructed data using
      the dataset's saved mean/std statistics.
   c. Convert reconstructed outputs to the requested evaluation representation
      ('absolute' or 'delta').
   d. Calculate the Mean Squared Error (MSE) between the original data and
      the reconstructed data in that representation.
3. Report the average MSE across the entire validation set.
4. (Optional) Generate and save plots comparing original vs. reconstructed
   trajectories for a few samples.
"""

import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import random

# Add project root to import custom modules
import sys
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import necessary classes from the training script
from TRAIN_RobotState_MAE import MAERobotStateModel, RobotStateEncoder

# Ensure reproducibility for visualization sampling
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


# ==============================================================================
# Modified RobotStateDataset for Evaluation
# ==============================================================================
class EvaluationRobotStateDataset(Dataset):
    """
    Modified dataset for evaluation. It returns both the model input
    (normalized, delta/absolute) and the original absolute window for comparison.
    """
    # Task ID mapping (same as training)
    TASK_MAPPING = {
        'Blue_point': 0,
        'Green_point': 1,
        'Red_point': 2,
        'White_point': 3,
        'Yellow_point': 4,
        'Eye_trocar': 5,
    }

    def __init__(self, root_dirs: list, window_size: int = 60, step: int = 10,
                 normalize: bool = True, data_representation: str = 'absolute',
                 provided_mean: torch.Tensor = None, provided_std: torch.Tensor = None):
        self.window_size = window_size
        self.step = step
        self.data_files = []
        self.normalize = normalize
        self.data_representation = data_representation
        self.provided_mean = provided_mean
        self.provided_std = provided_std

        print(f"Scanning for robot_states.npz in {root_dirs}...")
        for root_dir in root_dirs:
            for path in Path(root_dir).rglob('robot_states.npz'):
                self.data_files.append(path)
        print(f"Found {len(self.data_files)} robot_states.npz files.")

        self.windows = []
        all_data_for_norm = []
        for file_path in tqdm(self.data_files, desc="Creating windows"):
            try:
                data = np.load(file_path, mmap_mode='r')['robot_states']

                # Extract task ID from file path
                task_id = self._extract_task_id(file_path)

                if self.data_representation == 'delta':
                    delta_data = np.zeros_like(data)
                    delta_data[1:] = data[1:] - data[:-1]
                    data_to_process = delta_data
                else:
                    data_to_process = data

                if normalize:
                    all_data_for_norm.append(data_to_process)

                for i in range(0, len(data) - self.window_size + 1, self.step):
                    self.windows.append((file_path, i, task_id))
            except Exception as e:
                print(f"Warning: Could not load or process {file_path}: {e}")

        # Use provided statistics if available (from checkpoint)
        if self.provided_mean is not None and self.provided_std is not None:
            # Ensure mean/std are on CPU for DataLoader workers
            self.mean = self.provided_mean.cpu()
            self.std = self.provided_std.cpu()
            print(f"✅ Using provided normalization statistics (from checkpoint)")
        elif normalize and len(all_data_for_norm) > 0:
            all_data_concat = np.concatenate(all_data_for_norm, axis=0)
            self.mean = torch.from_numpy(all_data_concat.mean(axis=0).astype(np.float32))
            self.std = torch.from_numpy(all_data_concat.std(axis=0).astype(np.float32) + 1e-6)
            print(f"⚠️  WARNING: Using dataset-computed statistics (may cause incorrect results!)")
        else:
            self.mean = torch.zeros(12)
            self.std = torch.ones(12)

    def _extract_task_id(self, file_path: Path) -> int:
        """Extract task ID from file path."""
        path_str = str(file_path)
        for task_name, task_id in self.TASK_MAPPING.items():
            if task_name in path_str:
                return task_id
        # Default to 0 if no match found
        print(f"Warning: Could not extract task ID from {file_path}, defaulting to 0")
        return 0

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        file_path, start_idx, task_id = self.windows[idx]
        data = np.load(file_path, mmap_mode='r')['robot_states']

        original_absolute_window = torch.from_numpy(data[start_idx : start_idx + self.window_size].astype(np.float32))

        if self.data_representation == 'delta':
            model_input_window = torch.zeros_like(original_absolute_window)
            model_input_window[1:] = original_absolute_window[1:] - original_absolute_window[:-1]
        else:
            model_input_window = original_absolute_window.clone()

        if self.normalize:
            model_input_window = (model_input_window - self.mean) / self.std

        return {
            "model_input": model_input_window,
            "original_absolute": original_absolute_window,
            "task_id": task_id
        }


def compute_delta(sequence: torch.Tensor) -> torch.Tensor:
    """Return per-timestep deltas for a sequence (supports T x D or B x T x D)."""
    if sequence.dim() == 2:
        deltas = torch.zeros_like(sequence)
        deltas[1:] = sequence[1:] - sequence[:-1]
        return deltas
    if sequence.dim() == 3:
        deltas = torch.zeros_like(sequence)
        deltas[:, 1:] = sequence[:, 1:] - sequence[:, :-1]
        return deltas
    raise ValueError("Input tensor must be 2D or 3D for delta computation.")

# ==============================================================================
# Visualization Function
# ==============================================================================
def plot_trajectories(original, reconstructed, masked_indices, sample_idx, output_dir, representation):
    """Plots and saves the comparison of original and reconstructed trajectories, highlighting masked parts."""
    original = original.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    masked_indices = masked_indices.cpu().numpy()

    timesteps = np.arange(original.shape[0])
    mask = np.zeros(timesteps.shape, dtype=bool)
    if masked_indices.size > 0:
        mask[masked_indices] = True

    # Create figure with GridSpec layout
    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    fig.suptitle(f'Sample {sample_idx} - Reconstruction Comparison ({representation.capitalize()})', fontsize=18, y=0.98)

    # ========== 3D Trajectory (Left column, full height) ==========
    ax_3d = fig.add_subplot(gs[:, 0], projection='3d')

    # 1. Plot full original trajectory as a faint line
    ax_3d.plot(original[:, 6], original[:, 7], original[:, 8],
               label='Original Path', color='cyan', linewidth=1.5, alpha=0.6, zorder=1)

    # 2. Plot the ground truth points that were MASKED
    ax_3d.scatter(original[mask, 6], original[mask, 7], original[mask, 8],
                  label='Original (Masked)', color='gray', s=40, marker='x', alpha=0.8, zorder=3)

    # 3. Plot the reconstructed points at the MASKED locations
    ax_3d.scatter(reconstructed[mask, 6], reconstructed[mask, 7], reconstructed[mask, 8],
                  label='Reconstructed', color='red', s=50, marker='o', alpha=0.9, zorder=4, facecolors='none', edgecolors='red', linewidths=1.5)

    # 4. Plot the unmasked original points that were visible to the model
    ax_3d.scatter(original[~mask, 6], original[~mask, 7], original[~mask, 8],
                  label='Original (Visible)', color='blue', s=20, marker='.', alpha=0.8, zorder=2)

    # Mark start and end points
    ax_3d.scatter(original[0, 6], original[0, 7], original[0, 8],
                  color='green', s=120, marker='o', label='Start', zorder=5, edgecolors='black', linewidth=0.5)
    ax_3d.scatter(original[-1, 6], original[-1, 7], original[-1, 8],
                  color='purple', s=120, marker='s', label='End', zorder=5, edgecolors='black', linewidth=0.5)

    ax_3d.set_xlabel('X Position', fontsize=10)
    ax_3d.set_ylabel('Y Position', fontsize=10)
    ax_3d.set_zlabel('Z Position', fontsize=10)
    ax_3d.set_title('3D Trajectory', fontsize=14, pad=10)
    ax_3d.legend(loc='upper right', fontsize=9)
    ax_3d.grid(True, linestyle=':', alpha=0.5)

    # ========== XYZ & RPY Plots (Middle and Right columns) ==========
    plot_configs = [
        {'labels': ['Pos X', 'Pos Y', 'Pos Z'], 'indices': [6, 7, 8], 'col': 1},
        {'labels': ['Rot Roll', 'Rot Pitch', 'Rot Yaw'], 'indices': [9, 10, 11], 'col': 2}
    ]

    for config in plot_configs:
        axes = [fig.add_subplot(gs[i, config['col']]) for i in range(3)]
        for i, (ax, label) in enumerate(zip(axes, config['labels'])):
            dim_idx = config['indices'][i]

            # 1. Plot full original trajectory as a faint line
            ax.plot(timesteps, original[:, dim_idx], color='cyan', linewidth=1, alpha=0.7, label='_nolegend_')

            # 2. Plot unmasked original points
            ax.scatter(timesteps[~mask], original[~mask, dim_idx], label='Original (Visible)', color='blue', s=15, marker='.', alpha=0.8)

            # 3. Plot ground truth points that were MASKED
            # ax.scatter(timesteps[mask], original[mask, dim_idx], label='Original (Masked)', color='gray', s=30, marker='x', alpha=0.8)

            # 4. Plot reconstructed points at MASKED locations
            ax.scatter(timesteps[mask], reconstructed[mask, dim_idx], label='Reconstructed', color='red', s=40, marker='o', alpha=0.9, facecolors='none', edgecolors='red', linewidths=1)

            ax.set_ylabel(label, fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.5)
            if i == 2:
                ax.set_xlabel('Timestep', fontsize=10)

            # Set Y-lim and legend only once
            if i == 0:
                handles, labels = ax.get_legend_handles_labels()

    # Create a single legend for all 2D plots
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.68, 0.97), ncol=4, fontsize=12)

    save_path = os.path.join(output_dir, f'reconstruction_sample_{sample_idx}_{representation}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved visualization to {save_path}")


def plot_joint_angles(original, reconstructed, masked_indices, sample_idx, output_dir, representation):
    """Plots and saves the comparison of original and reconstructed joint angles, highlighting masked parts."""
    original = original.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    masked_indices = masked_indices.cpu().numpy()

    timesteps = np.arange(original.shape[0])
    mask = np.zeros(timesteps.shape, dtype=bool)
    if masked_indices.size > 0:
        mask[masked_indices] = True

    # Create figure with 3x2 layout for 6 joint angles
    fig, axs = plt.subplots(3, 2, figsize=(18, 12), sharex=True)
    fig.suptitle(f'Sample {sample_idx} - Joint Angles Reconstruction ({representation.capitalize()})', fontsize=18, y=0.99)

    joint_labels = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6']

    for i in range(6):
        row = i // 2
        col = i % 2
        ax = axs[row, col]

        # 1. Plot full original trajectory as a faint line
        ax.plot(timesteps, original[:, i], color='cyan', linewidth=1, alpha=0.7, label='_nolegend_')

        # 2. Plot unmasked original points
        ax.scatter(timesteps[~mask], original[~mask, i], label='Original (Visible)', color='blue', s=15, marker='.', alpha=0.8)

        # 3. Plot ground truth points that were MASKED
        ax.scatter(timesteps[mask], original[mask, i], label='Original (Masked)', color='gray', s=30, marker='o', alpha=0.8)

        # 4. Plot reconstructed points at MASKED locations
        ax.scatter(timesteps[mask], reconstructed[mask, i], label='Reconstructed', color='red', s=40, marker='.', alpha=0.9, facecolors='none', edgecolors='red', linewidths=1)

        ax.set_ylabel(joint_labels[i], fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.5)

        if row == 2:
            ax.set_xlabel('Timestep', fontsize=12)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=4, fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, f'reconstruction_sample_{sample_idx}_{representation}_joints.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved joint angles visualization to {save_path}")


def plot_3d_trajectory_only(original, reconstructed, masked_indices, sample_idx, output_dir, representation):
    """Plots and saves only the 3D trajectory comparison, highlighting masked parts."""
    original = original.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    masked_indices = masked_indices.cpu().numpy()

    timesteps = np.arange(original.shape[0])
    mask = np.zeros(timesteps.shape, dtype=bool)
    if masked_indices.size > 0:
        mask[masked_indices] = True

    # Create a large figure for just the 3D trajectory
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Plot full original trajectory as a faint line
    # ax.plot(original[:, 6], original[:, 7], original[:, 8],
    #         label='Original Path', color='cyan', linewidth=2, alpha=0.6, zorder=1)

    # 2. Plot the ground truth points that were MASKED
    ax.scatter(original[mask, 6], original[mask, 7], original[mask, 8],
               label='Original (Masked)', color='gray', s=50, marker='x', alpha=0.8, zorder=3)

    # 3. Plot the reconstructed points at the MASKED locations
    ax.scatter(reconstructed[mask, 6], reconstructed[mask, 7], reconstructed[mask, 8],
               label='Reconstructed', color='red', s=60, marker='o', alpha=0.9, zorder=4, facecolors='none', edgecolors='red', linewidths=1.5)

    # 4. Plot the unmasked original points that were visible to the model
    # ax.scatter(original[~mask, 6], original[~mask, 7], original[~mask, 8],
    #            label='Original (Visible)', color='blue', s=25, marker='.', alpha=0.8, zorder=2)

    # Mark start and end points
    ax.scatter(original[0, 6], original[0, 7], original[0, 8],
               color='green', s=150, marker='o', label='Start', zorder=5, edgecolors='black', linewidth=0.5)
    ax.scatter(original[-1, 6], original[-1, 7], original[-1, 8],
               color='purple', s=150, marker='s', label='End', zorder=5, edgecolors='black', linewidth=0.5)

    ax.set_xlabel('X Position', fontsize=12, labelpad=10)
    ax.set_ylabel('Y Position', fontsize=12, labelpad=10)
    ax.set_zlabel('Z Position', fontsize=12, labelpad=10)
    ax.set_title(f'Sample {sample_idx} - 3D Trajectory Reconstruction ({representation.capitalize()})',
                 fontsize=14, pad=20)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)

    # Set viewing angle for better visualization
    ax.view_init(elev=20, azim=45)

    save_path = os.path.join(output_dir, f'reconstruction_sample_{sample_idx}_{representation}_3d_trajectory.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved 3D trajectory visualization to {save_path}")


def main(args):
    """Main evaluation function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------------------------------------------------------------------------
    # 1. Load Checkpoint and Model Configuration
    # --------------------------------------------------------------------------
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint path not found: {args.checkpoint_path}")
        return

    print(f"Loading checkpoint from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    # In the training script, model args were not saved. We must use CLI args.
    print("Using command-line arguments for model configuration.")
    print("Please ensure these match the training configuration.")
    model_args = args

    # --------------------------------------------------------------------------
    # 2. Check if normalization was used during training
    # --------------------------------------------------------------------------
    use_normalization = checkpoint.get('normalize', True)  # Default True for old checkpoints
    normalization_mean = checkpoint.get('normalization_mean', None)
    normalization_std = checkpoint.get('normalization_std', None)

    if not use_normalization:
        print("✅ Model was trained WITHOUT normalization")
        print("   Evaluation will also skip normalization")
    else:
        print("✅ Model was trained WITH normalization")
        if normalization_mean is not None and normalization_std is not None:
            print("   Using saved normalization statistics from checkpoint")
            print(f"   Joints mean: {normalization_mean[:6]}")
            print(f"   Joints std: {normalization_std[:6]}")
            print(f"   Pose mean: {normalization_mean[6:]}")
            print(f"   Pose std: {normalization_std[6:]}")
        else:
            print("   ⚠️ WARNING: No normalization statistics found in checkpoint!")
            print("   This may cause incorrect evaluation results.")

    # --------------------------------------------------------------------------
    # 3. Setup Dataset with normalization
    # --------------------------------------------------------------------------
    full_dataset = EvaluationRobotStateDataset(
        root_dirs=args.dataset_paths,
        window_size=args.window_size,
        data_representation=args.data_representation,
        normalize=use_normalization,
        provided_mean=normalization_mean,
        provided_std=normalization_std
    )

    # We'll use a small, fixed portion for consistent evaluation
    val_size = int(len(full_dataset) * 0.05)
    train_size = len(full_dataset) - val_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size])

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False, # No need to shuffle for evaluation
        num_workers=args.num_workers,
        pin_memory=True
    )
    print(f"Evaluation dataset created with {len(val_dataset)} windows.")

    # --------------------------------------------------------------------------
    # 4. Setup Model
    # --------------------------------------------------------------------------
    encoder = RobotStateEncoder(
        temporal_length=model_args.window_size,
        model_dim=model_args.model_dim,
        output_dim=model_args.output_dim,
        num_heads=model_args.num_heads,
        num_layers=model_args.num_layers,
        use_fourier_features=model_args.use_fourier_features,
        num_frequencies=model_args.num_frequencies
    )
    model = MAERobotStateModel(
        encoder,
        decoder_dim=model_args.decoder_dim,
        num_tasks=model_args.num_tasks,
        task_embed_dim=model_args.task_embed_dim,
        decoder_num_layers=model_args.decoder_num_layers,
        decoder_num_heads=model_args.decoder_num_heads,
        decoder_dropout=model_args.decoder_dropout
    ).to(device)

    state_dict = checkpoint['model_state_dict']
    if all(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    # --- (Optional) Load Encoder from End-to-End model ---
    if args.load_encoder_from_e2e:
        print("\n" + "="*50)
        print(f"🔥 Overwriting encoder with weights from E2E model: {args.load_encoder_from_e2e}")
        e2e_checkpoint = torch.load(args.load_encoder_from_e2e, map_location=device)
        
        # The E2E checkpoint might contain the model within a 'model' or 'model_state_dict' key
        e2e_state_dict = e2e_checkpoint.get('model_state_dict', e2e_checkpoint)

        encoder_state_dict = {}
        # Handle DDP-trained models by checking for 'module.' prefix
        prefix_options = ['robot_state_encoder.', 'module.robot_state_encoder.']
        found_prefix = None

        for prefix in prefix_options:
            if any(k.startswith(prefix) for k in e2e_state_dict.keys()):
                found_prefix = prefix
                break
        
        if found_prefix:
            print(f"   Found encoder weights with prefix: '{found_prefix}'")
            for k, v in e2e_state_dict.items():
                if k.startswith(found_prefix):
                    encoder_state_dict[k.replace(found_prefix, '')] = v
        
        if not encoder_state_dict:
            print("⚠️ WARNING: No keys with 'robot_state_encoder.' or 'module.robot_state_encoder.' prefix found.")
            print("   The encoder weights were NOT overwritten.")
        else:
            missing_keys, unexpected_keys = model.encoder.load_state_dict(encoder_state_dict, strict=False)
            print("✅ Encoder weights loaded from E2E model.")
            if missing_keys:
                print(f"   - Missing keys in MAE encoder: {len(missing_keys)}")
            if unexpected_keys:
                print(f"   - Unexpected keys from E2E encoder: {len(unexpected_keys)}")
        print("="*50)

    model.eval()
    print("Model loaded successfully.")

    # --------------------------------------------------------------------------
    # 5. Run Evaluation Loop
    # --------------------------------------------------------------------------
    total_mse = 0.0
    samples_processed = 0
    samples_for_viz = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
            model_input = batch["model_input"].to(device)
            original_absolute = batch["original_absolute"].to(device)
            task_ids = batch["task_id"].to(device)
            B, T, D = model_input.shape

            # --- Masking (same as in training validation) ---
            num_masked = max(1, int(args.mask_ratio * T))
            masked_indices = torch.rand(B, T, device=device).topk(k=num_masked, dim=-1).indices
            masked_input = model_input.clone()
            batch_indices = torch.arange(B, device=device).unsqueeze(-1)
            mask_token = model.mask_token.expand(B, num_masked, D)
            masked_input.scatter_(1, masked_indices.unsqueeze(-1).expand(-1, -1, D), mask_token)

            # --- Forward Pass with task conditioning ---
            reconstructed = model(masked_input, task_ids)

            # --- Denormalization if needed ---
            if use_normalization and normalization_mean is not None and normalization_std is not None:
                # Denormalize reconstructed output
                mean = normalization_mean.to(device)
                std = normalization_std.to(device)
                reconstructed = reconstructed * std + mean

            # --- Obtain Absolute & Delta Representations ---
            original_delta = compute_delta(original_absolute)
            if args.data_representation == 'delta':
                # Reconstructed is in delta space (denormalized)
                reconstructed_delta = reconstructed
                initial_state = original_absolute[:, 0:1, :]
                reconstructed_absolute = initial_state + torch.cumsum(reconstructed_delta, dim=1)
            else:
                # Reconstructed is in absolute space (denormalized)
                reconstructed_absolute = reconstructed
                reconstructed_delta = compute_delta(reconstructed_absolute)

            if args.evaluation_representation == 'absolute':
                reconstruction_for_eval = reconstructed_absolute
                target_for_eval = original_absolute
            else:
                reconstruction_for_eval = reconstructed_delta
                target_for_eval = original_delta

            mse = torch.nn.functional.mse_loss(reconstruction_for_eval, target_for_eval)
            total_mse += mse.item() * B
            samples_processed += B

            # --- Save samples for visualization ---
            if i < (args.num_samples_to_visualize // args.batch_size) + 1:
                for j in range(min(args.batch_size, args.num_samples_to_visualize - len(samples_for_viz))):
                    samples_for_viz.append({
                        'original': target_for_eval[j],
                        'reconstructed': reconstruction_for_eval[j],
                        'masked_indices': masked_indices[j]
                    })

    # --------------------------------------------------------------------------
    # 6. Report Results and Visualize
    # --------------------------------------------------------------------------
    final_mse = total_mse / samples_processed
    print("\n" + "="*50)
    print(f"Evaluation Complete for model trained on '{args.data_representation}' inputs")
    print(f"Final {args.evaluation_representation.capitalize()} Reconstruction MSE: {final_mse:.6f}")
    print("="*50 + "\n")

    # Generate plots
    if args.num_samples_to_visualize > 0:
        print(f"Generating {len(samples_for_viz)} visualization plots...")
        for i, sample in enumerate(samples_for_viz):
            # Plot position and rotation (X,Y,Z + Roll,Pitch,Yaw)
            plot_trajectories(
                sample['original'],
                sample['reconstructed'],
                sample['masked_indices'],
                i,
                args.output_dir,
                args.evaluation_representation
            )
            # Plot joint angles (Joint 1-6)
            plot_joint_angles(
                sample['original'],
                sample['reconstructed'],
                sample['masked_indices'],
                i,
                args.output_dir,
                args.evaluation_representation
            )
            # Plot 3D trajectory separately
            plot_3d_trajectory_only(
                sample['original'],
                sample['reconstructed'],
                sample['masked_indices'],
                i,
                args.output_dir,
                args.evaluation_representation
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained Robot State MAE model.")

    # --- Required Arguments ---
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the model checkpoint file (e.g., robot_state_mae_best.pth).')
    parser.add_argument('--dataset_paths', type=str, nargs='*', required=True,
                        help='Paths to the root dataset directories used for validation.')
    parser.add_argument('--data_representation', type=str, choices=['absolute', 'delta'], required=True,
                        help="Data representation used during training. Must match the model.")
    parser.add_argument('--evaluation_representation', type=str, choices=['absolute', 'delta'], default='absolute',
                        help="Representation space in which to compute reconstruction metrics.")

    # --- Visualization Arguments ---
    parser.add_argument('--num_samples_to_visualize', type=int, default=5,
                        help='Number of trajectory samples to plot.')
    parser.add_argument('--output_dir', type=str, default='evaluation_results/mae_reconstruction',
                        help='Directory to save visualization plots.')

    # --- Model & Dataset Arguments (Required as they are not in the checkpoint) ---
    parser.add_argument('--window_size', type=int, default=100, help='Temporal window size for robot states.')
    parser.add_argument('--model_dim', type=int, default=512, help='Dimension of the Transformer model.')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of heads in the Transformer.')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of layers in the Transformer.')
    parser.add_argument('--output_dim', type=int, default=1024, help='Output dimension of the encoder projection head.')
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='Ratio of timesteps to mask during evaluation.')

    # --- Fourier Feature Arguments ---
    parser.add_argument('--use_fourier_features', action='store_true', default=True,
                        help='Use Fourier Feature Projection (default: True). Set --no-use_fourier_features to disable.')
    parser.add_argument('--no-use_fourier_features', dest='use_fourier_features', action='store_false',
                        help='Disable Fourier Feature Projection (use simple Linear projection).')
    parser.add_argument('--num_frequencies', type=int, default=8, help='Number of frequency bands for Fourier Features.')

    # --- Task Conditioning Arguments ---
    parser.add_argument('--num_tasks', type=int, default=6, help='Number of different tasks.')
    parser.add_argument('--task_embed_dim', type=int, default=64, help='Dimension of task embeddings.')

    # --- Decoder Architecture Arguments ---
    parser.add_argument('--decoder_dim', type=int, default=256, help='Dimension of the Transformer decoder.')
    parser.add_argument('--decoder_num_layers', type=int, default=4, help='Number of layers in the Transformer decoder.')
    parser.add_argument('--decoder_num_heads', type=int, default=8, help='Number of attention heads in the decoder.')
    parser.add_argument('--decoder_dropout', type=float, default=0.1, help='Dropout rate in the decoder.')

    # --- Dataloader Arguments ---
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers.')

    # --- Special Evaluation Arguments ---
    parser.add_argument('--load_encoder_from_e2e', type=str, default=None,
                        help='Path to an end-to-end model checkpoint to extract and evaluate its RobotStateEncoder.')

    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)
