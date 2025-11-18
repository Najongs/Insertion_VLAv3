#!/bin/bash
set -e

# =============================================================================
# VLA Training Pipeline with Advanced Sensor Encoder Pre-training
# =============================================================================
# This script orchestrates the full training pipeline, including:
# 1. Low-dimensional CLIP VLM cache generation for the sensor encoder.
# 2. Sensor encoder pre-training using a dual CLIP + Auxiliary Gate Loss.
# 3. Main VLA model training (Flow Matching) using the pre-trained encoders.
# =============================================================================

# --- Configuration ---
DATASET_PATHS_ALL=(
    # "/home/najo/NAS/VLA/dataset/New_dataset2/*_point"
    "/home/najo/NAS/VLA/dataset/New_dataset3/Red_point"
    "/home/najo/NAS/VLA/dataset/New_dataset4/Eye_trocar"
    "/home/najo/NAS/VLA/dataset/New_dataset5/Eye_trocar"
    "/home/najo/NAS/VLA/dataset/New_dataset6/Red_point"
    "/home/najo/NAS/VLA/dataset/New_dataset6/Blue_point"
    "/home/najo/NAS/VLA/dataset/New_dataset6/Green_point"
    "/home/najo/NAS/VLA/dataset/New_dataset6/White_point"
    "/home/najo/NAS/VLA/dataset/New_dataset6/Yellow_point"
)
CACHE_ROOT="/home/najo/NAS/VLA/dataset/cache"
CHECKPOINT_DIR="./checkpoints"
NUM_GPUS=4

# =============================================================================
# STEP 0: Pre-train Robot State Encoder (MAE)
# =============================================================================
echo ""
echo "=============== STEP 0: Pre-training Robot State Encoder (MAE) ==============="
echo "This is a crucial first step for the main VLA model."
echo ""

# First, convert all robot_states.csv to .npz for faster loading
# python convert_robot_states_to_npz.py "${DATASET_PATHS_ALL[@]}"

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
#     --nproc_per_node=$NUM_GPUS \
#     --master_port=29510 \
#     TRAIN_RobotState_MAE.py \
#     --epochs 200 \
#     --batch_size 128 \
#     --learning_rate 3e-4 \
#     --weight_decay 0.01 \
#     --dataset_paths "${DATASET_PATHS_ALL[@]}" \
#     --val_split 0.1 \
#     --window_size 100 \
#     --mask_ratio 0.75 \
#     --model_dim 512 \
#     --num_heads 8 \
#     --num_layers 6 \
#     --output_dim 1024 \
#     --min_lr 1e-6 \
#     --warmup_ratio 0.03 \
#     --num_workers 8 \
#     --checkpoint_dir "$CHECKPOINT_DIR"
#     --resume_from "$CHECKPOINT_DIR/robot_state_mae_latest.pth"

echo "=============== Robot State Encoder Pre-training Complete ==============="
echo ""

# =============================================================================
# STEP 1: Generate Low-Dimensional CLIP VLM Cache (Required)
# =============================================================================
echo ""
echo "=============== STEP 1: Generating Low-Dimensional CLIP VLM Cache ==============="
echo "Using VLM: Qwen/Qwen2.5-VL-3B-Instruct"
echo "Output Dimension: 512"
echo "This will take a while on the first run..."
echo ""

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
#     --nproc_per_node=$NUM_GPUS \
#     --master_port=29500 \
#     cache_clip_vlm_features.py \
#     --new_dataset_paths "${DATASET_PATHS_ALL[@]}" \
#     --cache_root "$CACHE_ROOT" \
#     --checkpoint_dir "$CHECKPOINT_DIR" \
#     --embedding_dim 512 \
#     --batch_size 4 \
#     --num_workers 4 \
#     --image_resize_height 360 \
#     --image_resize_width 640

echo "=============== CLIP VLM Cache Generation Complete ==============="
echo ""


# =============================================================================
# STEP 2: Pre-train Sensor Encoder with CLIP + Gate Loss (Required)
# =============================================================================
echo ""
echo "=============== STEP 2: Pre-training Sensor Encoder ==============="
echo "Using dual objective: CLIP Contrastive Loss + Auxiliary Gate Loss"
echo ""

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29501 \
    TRAIN_SensorImage_CLIP.py \
    --new_dataset_paths "${DATASET_PATHS_ALL[@]}" \
    --cache_root "$CACHE_ROOT" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --epochs 50 \
    --batch_size 64 \
    --num_workers 6 \
    --learning_rate 2e-4 \
    --embedding_dim 512 \
    --gate_loss_weight 0.25 \
    --contact_threshold 0.85

echo "=============== Sensor Encoder Pre-training Complete ==============="
echo ""


# =============================================================================
# STEP 3: Main VLA Training (Flow Matching)
# =============================================================================
echo ""
echo "=============== STEP 3: Main VLA Flow Matching Training ==============="
echo "Loading pre-trained sensor and robot state encoders."
echo "Using DeepSpeed ZeRO-2."
echo ""

# Note: This example trains on a single dataset for simplicity.
# For multi-dataset training, add more paths to --dataset_paths.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=$NUM_GPUS \
    TRAIN_FlowMatching.py \
    --deepspeed_config configs/deepspeed_zero2_offload.json \
    --dataset_paths "/home/najo/NAS/VLA/dataset/New_dataset6/*_point" \
    --epochs 50 \
    --batch_size 32 \
    --grad_accum 1 \
    --lr 1e-4 \
    --min_lr 1e-7 \
    --image_resize_height 360 \
    --image_resize_width 640 \
    --sensor_enabled \
    --num_workers 16 \
    --fusion_strategy "cross_attention" \
    --finetune_vl none \
    --val_split 0.05 \
    --load_robot_state_encoder_checkpoint "$CHECKPOINT_DIR/robot_state_mae_best.pth" \
    --load_sensor_encoder_checkpoint "$CHECKPOINT_DIR/sensor_clip_latest.pth" \
    --vlm_reuse_count 1 \
    --cache_root "$CACHE_ROOT/qwen_vl_features" \
    --skip_dataset_stats --use_cache --filter_by_cache

echo "=============== VLA Training Complete ==============="
echo ""

echo "✅✅✅ Full Training Pipeline Finished Successfully ✅✅✅"