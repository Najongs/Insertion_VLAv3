"""
Robot State Encoder Pre-training using Masked Auto-Encoding (MAE)

This script pre-trains the RobotStateEncoder on the task of reconstructing
masked portions of a robot state sequence.

Methodology:
1. A window of robot state data (joints + pose) is loaded.
2. A significant portion of the timesteps in the window are randomly masked.
3. The RobotStateEncoder processes the corrupted sequence.
4. A simple decoder head predicts the values of the original masked timesteps.
5. An MSE loss between the prediction and the ground truth is used for training.
6. This forces the encoder to learn the underlying dynamics and correlations
   of the robot's movement.
"""

import argparse
import os
import glob
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import math
from torch.optim.lr_scheduler import LambdaLR
import random

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import time
import wandb

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Add project root to import custom modules
import sys
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.unified_model import RobotStateEncoder

# =====================================
# 1. MAE Model Definition
# =====================================

class TransformerDecoder(nn.Module):
    """
    Transformer-based decoder for MAE reconstruction.
    Uses self-attention to model temporal dependencies.
    """
    def __init__(self, input_dim: int, decoder_dim: int, output_dim: int,
                 num_layers: int = 4, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.decoder_dim = decoder_dim
        self.output_dim = output_dim

        # Project encoder output + task embedding to decoder dimension
        self.input_projection = nn.Linear(input_dim, decoder_dim)

        # Transformer decoder layers
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=num_layers
        )

        # Output projection with residual connection
        self.output_projection = nn.Sequential(
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim) - concatenated encoder output + task embedding
        Returns:
            (B, T, output_dim) - reconstructed sequence
        """
        # Project to decoder dimension
        x = self.input_projection(x)  # (B, T, decoder_dim)

        # Apply transformer decoder
        x = self.transformer_decoder(x)  # (B, T, decoder_dim)

        # Project to output dimension
        x = self.output_projection(x)  # (B, T, output_dim)

        return x


class TaskConditioner(nn.Module):
    """
    Encapsulates task conditioning logic.
    Takes task IDs and produces a projected embedding for conditioning the
    input sequence and the raw embedding for the decoder.
    """
    def __init__(self, num_tasks: int, task_embed_dim: int, output_dim: int):
        super().__init__()
        self.task_embedding = nn.Embedding(num_tasks, task_embed_dim)
        self.task_projection = nn.Sequential(
            nn.Linear(task_embed_dim, output_dim),
            nn.GELU()
        )

    def forward(self, task_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            task_ids (torch.Tensor): (B,)
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - task_embed (B, task_embed_dim): Raw task embedding for decoder.
                - task_proj (B, output_dim): Projected embedding for encoder input.
        """
        task_embed = self.task_embedding(task_ids)
        task_proj = self.task_projection(task_embed)
        return task_embed, task_proj


class MAERobotStateModel(nn.Module):
    """
    Masked Auto-Encoder model built around the RobotStateEncoder.
    Task-Conditioned version: Uses task embeddings to condition the reconstruction.
    """
    def __init__(self, encoder: RobotStateEncoder, decoder_dim: int = 256,
                 num_tasks: int = 6, task_embed_dim: int = 64,
                 decoder_num_layers: int = 4, decoder_num_heads: int = 8,
                 decoder_dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.encoder_dim = encoder.model_dim
        self.decoder_dim = decoder_dim
        self.output_dim = encoder.input_dim  # Should be 12
        self.num_tasks = num_tasks
        self.task_embed_dim = task_embed_dim

        # Encapsulated task conditioning module
        self.task_conditioner = TaskConditioner(
            num_tasks=num_tasks,
            task_embed_dim=task_embed_dim,
            output_dim=self.output_dim
        )

        # Learnable mask token (B, T, D_in)로 브로드캐스트용
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.output_dim))

        # Transformer-based decoder for improved reconstruction
        self.decoder = TransformerDecoder(
            input_dim=self.encoder_dim + task_embed_dim,
            decoder_dim=decoder_dim,
            output_dim=self.output_dim,
            num_layers=decoder_num_layers,
            num_heads=decoder_num_heads,
            dropout=decoder_dropout
        )

    def forward(self, src: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src (torch.Tensor): Masked robot state sequence, shape (B, T, D_in)
            task_ids (torch.Tensor): Task IDs, shape (B,)

        Returns:
            torch.Tensor: Reconstructed sequence, shape (B, T, D_in)
        """
        B, T, D = src.shape

        # Get task embeddings and projected conditioning vector
        task_embed, task_proj = self.task_conditioner(task_ids)  # (B, task_embed_dim), (B, D_in)

        # Add task conditioning to input sequence
        # Broadcast task_proj across temporal dimension
        task_conditioned_src = src + task_proj.unsqueeze(1)  # (B, T, D_in)

        # Encode the masked sequence. The encoder itself is task-agnostic.
        encoded_sequence = self.encoder(task_conditioned_src, return_sequence=True)  # (B, T, D_encoder)

        # Concatenate raw task embedding to each timestep for the decoder
        task_embed_expanded = task_embed.unsqueeze(1).expand(-1, T, -1)  # (B, T, task_embed_dim)
        decoder_input = torch.cat([encoded_sequence, task_embed_expanded], dim=-1)  # (B, T, D_encoder + task_embed_dim)

        # Decode each timestep to reconstruct the original sequence
        reconstructed_sequence = self.decoder(decoder_input)  # (B, T, D_in)
        return reconstructed_sequence

# =====================================
# 1.5. Robot State Augmentation
# =====================================

class RobotStateAugmentation:
    """
    Robot state augmentation for MAE pre-training.
    All augmentations have ≤30% probability.
    """
    def __init__(self,
                 noise_std=0.005,
                 joint_scale_range=(0.98, 1.02),
                 pose_scale_range=(0.97, 1.03)):
        self.noise_std = noise_std
        self.joint_scale_range = joint_scale_range
        self.pose_scale_range = pose_scale_range

    def __call__(self, robot_states):
        """
        Args:
            robot_states: (T, 12) - 6 joints + 6 poses
        Returns:
            Augmented robot states
        """
        augmented = robot_states.clone()

        # 1. Gaussian noise (30% probability)
        if np.random.random() < 0.30:
            noise = torch.randn_like(augmented) * self.noise_std
            # Joint angles (0-5): smaller noise
            noise[:, :6] *= 0.2
            augmented += noise

        # 2. Magnitude scaling (30% probability)
        if np.random.random() < 0.30:
            # Joint scaling
            joint_scale = np.random.uniform(*self.joint_scale_range)
            augmented[:, :6] *= joint_scale

            # Pose scaling
            pose_scale = np.random.uniform(*self.pose_scale_range)
            augmented[:, 6:] *= pose_scale

        # 3. Time reversal (15% probability) - Disabled as it might be counter-intuitive
        # if np.random.random() < 0.15:
        #     augmented = torch.flip(augmented, dims=[0])

        return augmented


# =====================================
# 2. Robot State Dataset
# =====================================

class RobotStateDataset(Dataset):
    """
    Dataset that provides windows of robot state data from .npz files.
    Includes normalization for stable training.
    """
    # Task ID mapping
    TASK_MAPPING = {
        'Blue_point': 0,
        'Green_point': 1,
        'Red_point': 2,
        'White_point': 3,
        'Yellow_point': 4,
        'Eye_trocar': 5,
    }

    def __init__(self, root_dirs: list, window_size: int = 60, step: int = 10,
                 use_augmentation: bool = True, normalize: bool = True,
                 data_representation: str = 'absolute'):
        self.window_size = window_size
        self.step = step
        self.data_files = []
        self.normalize = normalize
        self.data_representation = data_representation

        # Data augmentation
        self.use_augmentation = use_augmentation
        self.is_training = True  # Training mode by default
        if self.use_augmentation:
            self.augmentation = RobotStateAugmentation()

        print(f"Scanning for robot_states.npz in {root_dirs}...")
        # Find all robot_states.npz files recursively
        for root_dir in root_dirs:
            for path in Path(root_dir).rglob('robot_states.npz'):
                self.data_files.append(path)

        print(f"Found {len(self.data_files)} robot_states.npz files.")

        self.windows = []
        all_data = []
        for file_path in tqdm(self.data_files, desc="Creating windows"):
            try:
                # Load the data using mmap_mode for memory efficiency
                data = np.load(file_path, mmap_mode='r')['robot_states']

                # Extract task ID from file path
                task_id = self._extract_task_id(file_path)

                if self.data_representation == 'delta':
                    delta_data = np.zeros_like(data)
                    delta_data[1:] = data[1:] - data[:-1]
                    # Assuming each file is one continuous episode.
                    # The first timestep's delta is 0.
                    data_to_process = delta_data
                else:
                    data_to_process = data

                # Collect data for computing normalization statistics
                if normalize:
                    all_data.append(data_to_process)

                # Create sliding windows from original data length with task ID
                for i in range(0, len(data) - self.window_size + 1, self.step):
                    self.windows.append((file_path, i, task_id))
            except Exception as e:
                print(f"Warning: Could not load or process {file_path}: {e}")

        # Compute normalization statistics
        if normalize and len(all_data) > 0:
            all_data_concat = np.concatenate(all_data, axis=0)
            self.mean = torch.from_numpy(all_data_concat.mean(axis=0).astype(np.float32))
            self.std = torch.from_numpy(all_data_concat.std(axis=0).astype(np.float32) + 1e-6)
            
            if self.data_representation == 'delta':
                print(f"📊 Using DELTA representation. Normalization statistics computed:")
            else:
                print(f"📊 Using ABSOLUTE representation. Normalization statistics computed:")
            print(f"   Joints mean: {self.mean[:6].numpy()}")
            print(f"   Joints std: {self.std[:6].numpy()}")
            print(f"   Pose mean: {self.mean[6:].numpy()}")
            print(f"   Pose std: {self.std[6:].numpy()}")
        else:
            self.mean = None
            self.std = None

    def _extract_task_id(self, file_path: Path) -> int:
        """Extract task ID from file path."""
        path_str = str(file_path)
        for task_name, task_id in self.TASK_MAPPING.items():
            if task_name in path_str:
                return task_id
        # Default to 0 if no match found
        print(f"Warning: Could not extract task ID from {file_path}, defaulting to 0")
        return 0

    def train(self):
        """Enable augmentation for training"""
        self.is_training = True

    def eval(self):
        """Disable augmentation for validation"""
        self.is_training = False

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        file_path, start_idx, task_id = self.windows[idx]

        # The file is re-opened here, which is fine with mmap
        data = np.load(file_path, mmap_mode='r')['robot_states']

        window = data[start_idx : start_idx + self.window_size]
        window_tensor = torch.from_numpy(window.astype(np.float32))

        # Convert to delta representation if needed
        if self.data_representation == 'delta':
            original_window = window_tensor.clone()
            delta_window = torch.zeros_like(original_window)
            delta_window[1:] = original_window[1:] - original_window[:-1]
            window_tensor = delta_window

        # Apply normalization
        if self.normalize and self.mean is not None and self.std is not None:
            window_tensor = (window_tensor - self.mean) / self.std

        # Apply augmentation (only during training)
        if self.use_augmentation and self.is_training:
            window_tensor = self.augmentation(window_tensor)

        return {
            'robot_states': window_tensor,
            'task_id': task_id
        }

# =====================================
# 3. Main Training Function
# =====================================

def build_trapezoid_scheduler(
    optimizer,
    total_steps: int,
    *,
    base_lr: float = 1e-4,
    min_lr: float = 1e-6,
    warmup_ratio: float = 0.03,
    hold_ratio: float = 0.02,
):
    """LLM 스타일: Warmup -> Hold -> Cosine Decay"""
    warmup_steps = int(total_steps * warmup_ratio)
    hold_steps = int(total_steps * hold_ratio)
    decay_steps = max(1, total_steps - warmup_steps - hold_steps)
    floor = min_lr / max(base_lr, 1e-12)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        elif step < warmup_steps + hold_steps:
            return 1.0
        else:
            t = (step - warmup_steps - hold_steps) / decay_steps
            cos_val = 0.5 * (1.0 + math.cos(math.pi * t))
            return floor + (1.0 - floor) * cos_val

    sched = LambdaLR(optimizer, lr_lambda=lr_lambda)
    prev_lr = base_lr * lr_lambda(0)
    for g in optimizer.param_groups:
        g["lr"] = prev_lr
    return sched


def compute_weighted_mse_loss(pred, target, mask, joint_weight=1.0, position_weight=5.0, rotation_weight=1.0):
    """
    Compute weighted MSE loss for robot state reconstruction.

    Loss is computed separately for three components with different weights:
    - Joints (0-5): Joint angles
    - Position (6-8): X, Y, Z coordinates (typically higher weight for precise control)
    - Rotation (9-11): Roll, Pitch, Yaw angles

    Args:
        pred: (B, T, 12) predicted robot states
        target: (B, T, 12) ground truth robot states
        mask: (B, T, 12) boolean mask for masked positions
        joint_weight: weight for joint angles (dimensions 0-5)
        position_weight: weight for position (dimensions 6-8: x, y, z)
        rotation_weight: weight for rotation (dimensions 9-11: roll, pitch, yaw)

    Returns:
        Tuple of (total_loss, loss_joints, loss_position, loss_rotation)
    """
    # Separate into three components
    pred_joints = pred[..., :6]      # Joint angles
    pred_position = pred[..., 6:9]   # X, Y, Z position
    pred_rotation = pred[..., 9:12]  # Roll, Pitch, Yaw

    target_joints = target[..., :6]
    target_position = target[..., 6:9]
    target_rotation = target[..., 9:12]

    mask_joints = mask[..., :6]
    mask_position = mask[..., 6:9]
    mask_rotation = mask[..., 9:12]

    device = pred.device
    zero = torch.zeros((), device=device)

    # Compute individual losses
    if mask_joints.any():
        loss_joints = F.mse_loss(pred_joints[mask_joints], target_joints[mask_joints])
    else:
        loss_joints = zero

    if mask_position.any():
        loss_position = F.mse_loss(pred_position[mask_position], target_position[mask_position])
    else:
        loss_position = zero

    if mask_rotation.any():
        loss_rotation = F.mse_loss(pred_rotation[mask_rotation], target_rotation[mask_rotation])
    else:
        loss_rotation = zero

    total_loss = (joint_weight * loss_joints +
                  position_weight * loss_position +
                  rotation_weight * loss_rotation)

    return total_loss, loss_joints, loss_position, loss_rotation


def main(args):
    # DDP Setup
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    is_main_process = local_rank == 0

    if is_main_process:
        print(f"Using {torch.cuda.device_count()} GPUs for MAE pre-training.")
        print(f"Loss weights - Joints: {args.joint_weight}, Position (X,Y,Z): {args.position_weight}, Rotation (R,P,Y): {args.rotation_weight}")

    # Model Setup
    encoder = RobotStateEncoder(
        temporal_length=args.window_size,
        model_dim=args.model_dim,
        output_dim=args.output_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    )
    model = MAERobotStateModel(
        encoder,
        decoder_dim=args.decoder_dim,
        num_tasks=args.num_tasks,
        task_embed_dim=args.task_embed_dim,
        decoder_num_layers=args.decoder_num_layers,
        decoder_num_heads=args.decoder_num_heads,
        decoder_dropout=args.decoder_dropout
    ).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Dataset and DataLoader
    if is_main_process:
        print("Creating dataset...")

    # Load dataset from multiple paths
    if is_main_process:
        print(f"Dataset paths received: {args.dataset_paths}")

    try:
        dataset = RobotStateDataset(
            root_dirs=args.dataset_paths,
            window_size=args.window_size,
            data_representation=args.data_representation,
            normalize=True  # Enable normalization for stable training
        )
        if len(dataset) == 0:
            raise ValueError("No data found in the provided dataset paths.")
    except Exception as e:
        raise ValueError(f"Error loading dataset: {e}")

    # Split dataset into training and validation
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,                 # 각 rank 배치 수 정렬
        persistent_workers=(args.num_workers > 0)
    )
    val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )

    if is_main_process:
        print(f"Dataset created with {len(train_dataset)} training windows and {len(val_dataset)} validation windows.")

    # Scheduler
    total_steps = (len(train_loader) * args.epochs) // args.grad_accum
    scheduler = build_trapezoid_scheduler(
        optimizer, total_steps=total_steps, base_lr=args.learning_rate, min_lr=args.min_lr,
        warmup_ratio=args.warmup_ratio, hold_ratio=args.hold_ratio,
    )

    # Checkpoint Loading
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume_from and os.path.exists(args.resume_from):
        if is_main_process:
            print(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)

        # Filter out projection layer if output_dim changed
        state_dict = checkpoint['model_state_dict']
        filtered_state_dict = {}
        skipped_keys = []
        for k, v in state_dict.items():
            # Skip encoder.projection layers (MAE reconstruction head, not used in downstream)
            if 'encoder.projection' in k:
                skipped_keys.append(k)
                continue
            filtered_state_dict[k] = v

        if skipped_keys and is_main_process:
            print(f"⚠️  Skipped loading projection layers (will be re-initialized): {len(skipped_keys)} keys")

        # Load with strict=False to allow missing projection keys
        missing_keys, unexpected_keys = model.module.load_state_dict(filtered_state_dict, strict=False)
        if is_main_process and (missing_keys or unexpected_keys):
            print(f"   Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")

        if args.reset_lr:
            if is_main_process:
                print(f"🔄 Resetting learning rate to {args.learning_rate} and scheduler (keeping model weights)")
        else:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']

    # Initialize wandb (only on main process)
    if is_main_process:
        wandb.init(
            project="QwenVLA-RobotStateMAE",
            name=f"robot_state_mae_{time.strftime('%m%d_%H%M')}",
            resume="allow",
            id=f"robot_state_mae_{int(time.time())}",
            settings=wandb.Settings(start_method="thread", _disable_stats=True),
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "window_size": args.window_size,
                "mask_ratio": args.mask_ratio,
                "model_dim": args.model_dim,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "output_dim": args.output_dim,
                "num_tasks": args.num_tasks,
                "task_embed_dim": args.task_embed_dim,
                "decoder_dim": args.decoder_dim,
                "decoder_num_layers": args.decoder_num_layers,
                "decoder_num_heads": args.decoder_num_heads,
                "decoder_dropout": args.decoder_dropout,
                "joint_weight": args.joint_weight,
                "position_weight": args.position_weight,
                "rotation_weight": args.rotation_weight,
                "data_representation": args.data_representation,
            }
        )

    global_step = 0

    # Training Loop
    if is_main_process:
        print("Starting MAE pre-training...")
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        print(f"Checkpoints will be saved to: {args.checkpoint_dir}")

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()

        # Enable augmentation for training
        # The dataset object from random_split is a Subset, we need to access its underlying dataset
        if hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'train'):
            train_dataset.dataset.train()

        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]", disable=not is_main_process)

        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_progress_bar, start=1):
            original_data = batch['robot_states'].to(device, non_blocking=True)
            task_ids = batch['task_id'].to(device, non_blocking=True)
            B, T, D = original_data.shape

            # Create mask
            num_masked = max(1, int(args.mask_ratio * T))  # 최소 1
            masked_indices = torch.rand(original_data.shape[:2], device=device).topk(k=num_masked, dim=-1).indices

            masked_input = original_data.clone()
            loss_mask = torch.zeros_like(original_data, dtype=torch.bool)

            batch_indices = torch.arange(B, device=device).unsqueeze(-1)
            mask_tok = model.module.mask_token.expand(B, masked_indices.shape[1], original_data.shape[-1]).to(device)

            # gather 방식으로 치환 위치에만 mask token 주입
            masked_input[batch_indices, masked_indices] = mask_tok
            loss_mask[batch_indices, masked_indices] = True

            # Forward pass with task conditioning
            reconstructed_data = model(masked_input, task_ids)
            loss, loss_joints, loss_position, loss_rotation = compute_weighted_mse_loss(
                reconstructed_data, original_data, loss_mask,
                joint_weight=args.joint_weight,
                position_weight=args.position_weight,
                rotation_weight=args.rotation_weight
            )

            (loss / args.grad_accum).backward()
            if step % args.grad_accum == 0:
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None and args.sched_on == "step":
                    scheduler.step()

                global_step += 1

            if is_main_process:
                current_lr = optimizer.param_groups[0]['lr']
                j_loss_val = loss_joints if isinstance(loss_joints, float) else loss_joints.item()
                pos_loss_val = loss_position if isinstance(loss_position, float) else loss_position.item()
                rot_loss_val = loss_rotation if isinstance(loss_rotation, float) else loss_rotation.item()

                train_progress_bar.set_postfix(
                    loss=loss.item(),
                    j_loss=j_loss_val,
                    pos_loss=pos_loss_val,
                    rot_loss=rot_loss_val,
                    lr=f"{current_lr:.2e}"
                )

                # Log to wandb
                if step % args.grad_accum == 0:  # Only log after optimizer step
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/joint_loss": j_loss_val,
                        "train/position_loss": pos_loss_val,
                        "train/rotation_loss": rot_loss_val,
                        "train/lr": current_lr,
                        "global_step": global_step,
                        "epoch": epoch + 1
                    })

        # 에폭 스케줄러는 에폭당 1회만
        if scheduler is not None and args.sched_on == "epoch":
            scheduler.step()

        # Validation Loop
        model.eval()

        # Disable augmentation for validation
        # The dataset object from random_split is a Subset, we need to access its underlying dataset
        if hasattr(val_dataset, 'dataset') and hasattr(val_dataset.dataset, 'eval'):
            val_dataset.dataset.eval()

        total_val_loss = 0
        total_val_joint_loss = 0
        total_val_position_loss = 0
        total_val_rotation_loss = 0
        val_count = 0
        with torch.no_grad():
            val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]", disable=not is_main_process)
            for batch in val_progress_bar:
                original_data = batch['robot_states'].to(device, non_blocking=True)
                task_ids = batch['task_id'].to(device, non_blocking=True)
                B, T, D = original_data.shape

                # In validation, we can reconstruct the whole sequence to check general performance
                # or stick to the same masking strategy. Sticking to masking is a better test.
                num_masked = max(1, int(args.mask_ratio * T))
                masked_indices = torch.rand(original_data.shape[:2], device=device).topk(k=num_masked, dim=-1).indices

                masked_input = original_data.clone()
                loss_mask = torch.zeros_like(original_data, dtype=torch.bool)

                batch_indices = torch.arange(B, device=device).unsqueeze(-1)
                # Use mask_token for consistency with training
                mask_tok = model.module.mask_token.expand(B, masked_indices.shape[1], original_data.shape[-1]).to(device)
                masked_input[batch_indices, masked_indices] = mask_tok
                loss_mask[batch_indices, masked_indices] = True

                reconstructed_data = model(masked_input, task_ids)
                val_loss, val_loss_joints, val_loss_position, val_loss_rotation = compute_weighted_mse_loss(
                    reconstructed_data, original_data, loss_mask,
                    joint_weight=args.joint_weight,
                    position_weight=args.position_weight,
                    rotation_weight=args.rotation_weight
                )

                total_val_loss += val_loss.item() * B
                total_val_joint_loss += val_loss_joints.item() * B
                total_val_position_loss += val_loss_position.item() * B
                total_val_rotation_loss += val_loss_rotation.item() * B
                val_count += B

        tensor = torch.tensor([total_val_loss, total_val_joint_loss, total_val_position_loss, total_val_rotation_loss, val_count], device=device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        global_val_loss, global_val_joint_loss, global_val_position_loss, global_val_rotation_loss, global_count = tensor.tolist()
        avg_val_loss = global_val_loss / max(1.0, global_count)
        avg_val_joint_loss = global_val_joint_loss / max(1.0, global_count)
        avg_val_position_loss = global_val_position_loss / max(1.0, global_count)
        avg_val_rotation_loss = global_val_rotation_loss / max(1.0, global_count)

        if is_main_process:
            print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f}, Joint: {avg_val_joint_loss:.4f}, Position: {avg_val_position_loss:.4f}, Rotation: {avg_val_rotation_loss:.4f}")

            # Log validation metrics to wandb
            wandb.log({
                "val/loss": avg_val_loss,
                "val/joint_loss": avg_val_joint_loss,
                "val/position_loss": avg_val_position_loss,
                "val/rotation_loss": avg_val_rotation_loss,
                "epoch": epoch + 1
            })

            # Save checkpoint
            latest_checkpoint_path = os.path.join(args.checkpoint_dir, "robot_state_mae_latest.pth")
            best_checkpoint_path = os.path.join(args.checkpoint_dir, "robot_state_mae_best.pth")

            # Get normalization statistics from the dataset
            # The dataset object from random_split is a Subset, we need to access its underlying dataset
            underlying_dataset = train_dataset.dataset
            normalization_mean = underlying_dataset.mean if hasattr(underlying_dataset, 'mean') else None
            normalization_std = underlying_dataset.std if hasattr(underlying_dataset, 'std') else None

            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),  # Full model (encoder + decoder) for resuming training
                'encoder_state_dict': model.module.encoder.state_dict(),  # Encoder only for downstream tasks
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'loss': loss.item(),
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
                'data_representation': args.data_representation,  # Save representation type
                'normalize': True,  # Mark that normalization was used
                'normalization_mean': normalization_mean,  # Save mean for evaluation
                'normalization_std': normalization_std,  # Save std for evaluation
            }

            # Save best checkpoint if validation loss improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_data['best_val_loss'] = best_val_loss
                torch.save(checkpoint_data, best_checkpoint_path)
                print(f"✨ New best model saved with validation loss: {avg_val_loss:.4f}")
            
            # Always save the latest checkpoint
            torch.save(checkpoint_data, latest_checkpoint_path)
            if is_main_process:
                print(f"💾 Latest model saved to {latest_checkpoint_path}")

    if is_main_process:
        print("MAE Pre-training finished.")
        wandb.finish()

    dist.destroy_process_group()

# =====================================
# 4. Argparse and Entrypoint
# =====================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train RobotStateEncoder with MAE.")

    # Dataset & Dataloader
    parser.add_argument('--dataset_paths', type=str, nargs='+', required=True,
                        help='Paths to the root dataset directories.')
    parser.add_argument('--data_representation', type=str, choices=['absolute', 'delta'], default='absolute',
                        help='Representation of robot state data: absolute positions or delta velocities.')
    parser.add_argument('--window_size', type=int, default=100, help='Temporal window size for robot states.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size per GPU.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of dataloader workers per GPU.')
    parser.add_argument('--val_split', type=float, default=0.05, help='Proportion of the dataset to use for validation.')

    # Model & Architecture
    parser.add_argument('--model_dim', type=int, default=512, help='Dimension of the Transformer model.')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of heads in the Transformer.')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of layers in the Transformer.')
    parser.add_argument('--output_dim', type=int, default=1024, help='Output dimension of the encoder projection head.')
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='Ratio of timesteps to mask.')

    # Task Conditioning
    parser.add_argument('--num_tasks', type=int, default=6, help='Number of different tasks (Blue_point, Green_point, Red_point, White_point, Yellow_point, Eye_trocar).')
    parser.add_argument('--task_embed_dim', type=int, default=64, help='Dimension of task embeddings.')

    # Decoder Architecture
    parser.add_argument('--decoder_dim', type=int, default=256, help='Dimension of the Transformer decoder.')
    parser.add_argument('--decoder_num_layers', type=int, default=4, help='Number of layers in the Transformer decoder.')
    parser.add_argument('--decoder_num_heads', type=int, default=8, help='Number of attention heads in the decoder.')
    parser.add_argument('--decoder_dropout', type=float, default=0.1, help='Dropout rate in the decoder.')

    # Training & Optimization
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for AdamW optimizer.')
    parser.add_argument('--grad_accum', type=int, default=1, help='Number of gradient accumulation steps.')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate.')
    parser.add_argument('--warmup_ratio', type=float, default=0.03, help='Ratio of total steps for learning rate warmup.')
    parser.add_argument('--hold_ratio', type=float, default=0.02, help='Ratio of total steps for holding learning rate after warmup.')
    parser.add_argument('--sched_on', type=str, choices=['step', 'epoch'], default='step', help="When to step the scheduler: 'step' or 'epoch'.")

    # Loss Weights
    parser.add_argument('--joint_weight', type=float, default=0.5, help='Weight for joint angle reconstruction loss.')
    parser.add_argument('--position_weight', type=float, default=5.0, help='Weight for position (x,y,z) reconstruction loss.')
    parser.add_argument('--rotation_weight', type=float, default=0.5, help='Weight for rotation (roll,pitch,yaw) reconstruction loss.')

    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='/home/najo/NAS/VLA/Insertion_VLAv2/checkpoints', help='Directory to save checkpoints.')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to a checkpoint to resume training from.')
    parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate and scheduler when resuming from checkpoint.')

    args = parser.parse_args()
    main(args)
