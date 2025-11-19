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
    def __init__(self, root_dir: str, window_size: int = 60, step: int = 10,
                 normalize: bool = True, data_representation: str = 'absolute'):
        self.window_size = window_size
        self.step = step
        self.data_files = []
        self.normalize = normalize
        self.data_representation = data_representation

        print(f"Scanning for robot_states.npz in {root_dir}...")
        for path in Path(root_dir).rglob('robot_states.npz'):
            self.data_files.append(path)
        print(f"Found {len(self.data_files)} robot_states.npz files.")

        self.windows = []
        all_data_for_norm = []
        for file_path in tqdm(self.data_files, desc="Creating windows"):
            try:
                data = np.load(file_path, mmap_mode='r')['robot_states']
                if self.data_representation == 'delta':
                    delta_data = np.zeros_like(data)
                    delta_data[1:] = data[1:] - data[:-1]
                    data_to_process = delta_data
                else:
                    data_to_process = data

                if normalize:
                    all_data_for_norm.append(data_to_process)

                for i in range(0, len(data) - self.window_size + 1, self.step):
                    self.windows.append((file_path, i))
            except Exception as e:
                print(f"Warning: Could not load or process {file_path}: {e}")

        if normalize and len(all_data_for_norm) > 0:
            all_data_concat = np.concatenate(all_data_for_norm, axis=0)
            self.mean = torch.from_numpy(all_data_concat.mean(axis=0).astype(np.float32))
            self.std = torch.from_numpy(all_data_concat.std(axis=0).astype(np.float32) + 1e-6)
        else:
            self.mean = torch.zeros(12)
            self.std = torch.ones(12)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        file_path, start_idx = self.windows[idx]
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
            "original_absolute": original_absolute_window
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
def plot_trajectories(original, reconstructed, sample_idx, output_dir, representation):
    """Plots and saves the comparison of original and reconstructed trajectories."""
    original = original.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    
    timesteps = np.arange(original.shape[0])
    
    fig, axs = plt.subplots(3, 2, figsize=(18, 12))
    fig.suptitle(f'Sample {sample_idx} - Reconstruction Comparison ({representation.capitalize()})', fontsize=16)

    # Plotting XYZ positions
    labels_xyz = ['Pos X', 'Pos Y', 'Pos Z']
    for i in range(3):
        ax = axs[i, 0]
        ax.plot(timesteps, original[:, 6+i], label='Original', color='blue')
        ax.plot(timesteps, reconstructed[:, 6+i], label='Reconstructed', color='red', linestyle='--')
        ax.set_ylabel(labels_xyz[i])
        ax.legend()
        ax.grid(True, linestyle=':')

    # Plotting Roll, Pitch, Yaw rotations
    labels_rpy = ['Rot Roll', 'Rot Pitch', 'Rot Yaw']
    for i in range(3):
        ax = axs[i, 1]
        ax.plot(timesteps, original[:, 9+i], label='Original', color='blue')
        ax.plot(timesteps, reconstructed[:, 9+i], label='Reconstructed', color='red', linestyle='--')
        ax.set_ylabel(labels_rpy[i])
        ax.legend()
        ax.grid(True, linestyle=':')

    for ax in axs.flat:
        ax.set_xlabel('Timestep')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, f'reconstruction_sample_{sample_idx}_{representation}.png')
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved visualization to {save_path}")


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
    # 2. Setup Model
    # --------------------------------------------------------------------------
    encoder = RobotStateEncoder(
        temporal_length=model_args.window_size,
        model_dim=model_args.model_dim,
        output_dim=model_args.output_dim,
        num_heads=model_args.num_heads,
        num_layers=model_args.num_layers
    )
    model = MAERobotStateModel(encoder).to(device)

    state_dict = checkpoint['model_state_dict']
    if all(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully.")

    # --------------------------------------------------------------------------
    # 3. Setup Dataset and DataLoader
    # --------------------------------------------------------------------------
    datasets = []
    for dataset_path in args.dataset_paths:
        ds = EvaluationRobotStateDataset(
            root_dir=dataset_path,
            window_size=args.window_size,
            data_representation=args.data_representation
        )
        datasets.append(ds)
    
    full_dataset = ConcatDataset(datasets)
    
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

    # Get normalization stats from the dataset
    # Assuming all concatenated datasets have similar stats, we use the first one.
    mean = datasets[0].mean.to(device)
    std = datasets[0].std.to(device)

    # --------------------------------------------------------------------------
    # 4. Run Evaluation Loop
    # --------------------------------------------------------------------------
    total_mse = 0.0
    samples_processed = 0
    samples_for_viz = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
            model_input = batch["model_input"].to(device)
            original_absolute = batch["original_absolute"].to(device)
            B, T, D = model_input.shape

            # --- Masking (same as in training validation) ---
            num_masked = max(1, int(args.mask_ratio * T))
            masked_indices = torch.rand(B, T, device=device).topk(k=num_masked, dim=-1).indices
            masked_input = model_input.clone()
            batch_indices = torch.arange(B, device=device).unsqueeze(-1)
            mask_token = model.mask_token.expand(B, num_masked, D)
            masked_input.scatter_(1, masked_indices.unsqueeze(-1).expand(-1, -1, D), mask_token)

            # --- Forward Pass ---
            reconstructed_normalized = model(masked_input)

            # --- De-normalization ---
            reconstructed_unnormalized = reconstructed_normalized * std + mean

            # --- Obtain Absolute & Delta Representations ---
            original_delta = compute_delta(original_absolute)
            if args.data_representation == 'delta':
                reconstructed_delta = reconstructed_unnormalized
                initial_state = original_absolute[:, 0:1, :]
                reconstructed_absolute = initial_state + torch.cumsum(reconstructed_delta, dim=1)
            else:
                reconstructed_absolute = reconstructed_unnormalized
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
                        'reconstructed': reconstruction_for_eval[j]
                    })

    # --------------------------------------------------------------------------
    # 5. Report Results and Visualize
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
            plot_trajectories(
                sample['original'],
                sample['reconstructed'],
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

    # --- Dataloader Arguments ---
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers.')

    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)
