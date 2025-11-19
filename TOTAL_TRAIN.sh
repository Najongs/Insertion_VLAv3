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
    # "/home/najo/NAS/VLA/dataset/New_dataset3/Red_point"
    # "/home/najo/NAS/VLA/dataset/New_dataset4/Eye_trocar"
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

# Sensor CLIP Pre-training specific configuration
SENSOR_CLIP_RUN_NAME="sensor_clip_pretrain_$(date +%Y%m%d_%H%M%S)" # Unique name for this run

# =============================================================================
# STEP 0: Pre-train Robot State Encoder (MAE)
# =============================================================================
echo ""
echo "=============== STEP 0: Pre-training Robot State Encoder (MAE) ==============="
echo "This is a crucial first step for the main VLA model."
echo "We will train two separate models: one for 'delta' and one for 'absolute' representations."
echo ""

# First, convert all robot_states.csv to .npz for faster loading
# python convert_robot_states_to_npz.py "${DATASET_PATHS_ALL[@]}"

# --- Train 'delta' representation model ---
# echo "--- Training MAE with 'delta' representation ---"
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
#     --nproc_per_node=4 \
#     --master_port=29515 \
#     TRAIN_RobotState_MAE.py \
#     --epochs 100 \
#     --batch_size 128 \
#     --learning_rate 1e-4 \
#     --weight_decay 0.01 \
#     --dataset_paths "/home/najo/NAS/VLA/dataset/New_dataset3/Red_point" "/home/najo/NAS/VLA/dataset/New_dataset4/Eye_trocar" "/home/najo/NAS/VLA/dataset/New_dataset5/Eye_trocar" "/home/najo/NAS/VLA/dataset/New_dataset6/Red_point" "/home/najo/NAS/VLA/dataset/New_dataset6/Blue_point" "/home/najo/NAS/VLA/dataset/New_dataset6/Green_point" "/home/najo/NAS/VLA/dataset/New_dataset6/White_point" "/home/najo/NAS/VLA/dataset/New_dataset6/Yellow_point" \   --val_split 0.1 \
#     --window_size 100 \
#     --mask_ratio 0.75 \
#     --model_dim 512 \
#     --num_heads 8 \
#     --num_layers 6 \
#     --output_dim 1024 \
#     --min_lr 1e-7 \
#     --num_workers 4 \
#     --checkpoint_dir "./checkpoints" \
#     --data_representation "delta"

# # Rename the best model checkpoint
# if [ -f "$CHECKPOINT_DIR/robot_state_mae_best.pth" ]; then
#     mv "$CHECKPOINT_DIR/robot_state_mae_best.pth" "$CHECKPOINT_DIR/robot_state_mae_best_delta.pth"
#     echo "✅ Renamed best delta model to robot_state_mae_best_delta.pth"
# fi
# if [ -f "$CHECKPOINT_DIR/robot_state_mae_latest.pth" ]; then
#     mv "$CHECKPOINT_DIR/robot_state_mae_latest.pth" "$CHECKPOINT_DIR/robot_state_mae_latest_delta.pth"
# fi


# --- Train 'absolute' representation model ---
# echo ""
# echo "--- Training MAE with 'absolute' representation ---"
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
#     --nproc_per_node=$NUM_GPUS \
#     --master_port=29510 \
#     TRAIN_RobotState_MAE.py \
#     --epochs 100 \
#     --batch_size 128 \
#     --learning_rate 1e-4 \
#     --weight_decay 0.01 \
#     --dataset_paths "${DATASET_PATHS_ALL[@]}" \
#     --val_split 0.1 \
#     --window_size 100 \
#     --mask_ratio 0.75 \
#     --model_dim 512 \
#     --num_heads 8 \
#     --num_layers 6 \
#     --output_dim 1024 \
#     --min_lr 1e-7 \
#     --num_workers 4 \
#     --checkpoint_dir "$CHECKPOINT_DIR" \
#     --data_representation "absolute"

# # Rename the best model checkpoint
# if [ -f "$CHECKPOINT_DIR/robot_state_mae_best.pth" ]; then
#     mv "$CHECKPOINT_DIR/robot_state_mae_best.pth" "$CHECKPOINT_DIR/robot_state_mae_best_absolute.pth"
#     echo "✅ Renamed best absolute model to robot_state_mae_best_absolute.pth"
# fi
# if [ -f "$CHECKPOINT_DIR/robot_state_mae_latest.pth" ]; then
#     mv "$CHECKPOINT_DIR/robot_state_mae_latest.pth" "$CHECKPOINT_DIR/robot_state_mae_latest_absolute.pth"
# fi

# echo "=============== Robot State Encoder Pre-training Complete ==============="
# echo ""


# =============================================================================
# STEP 0.5: Evaluate Robot State Encoder (MAE)
echo ""
echo "=============== STEP 0.5: Evaluating Robot State Encoders ==============="
echo "Running both absolute/delta reconstruction metrics for each model."
echo ""

# # --- Evaluate 'absolute' model ---
# echo "--- Evaluating 'absolute' representation model (absolute-space metric) ---"
# python evaluate_robot_state_mae.py \
#     --checkpoint_path "./checkpoints/robot_state_mae_best_absolute.pth" \
#     --data_representation absolute \
#     --evaluation_representation absolute \
#     --dataset_paths "/home/najo/NAS/VLA/dataset/New_dataset6/Yellow_point" \
#     --output_dir "evaluation_results/mae_reconstruction_absolute" \
#     --window_size 100 \
#     --model_dim 512 \
#     --num_heads 8 \
#     --num_layers 6 \
#     --output_dim 1024

# echo "--- Evaluating 'absolute' representation model (delta-space metric) ---"
# python evaluate_robot_state_mae.py \
#     --checkpoint_path "./checkpoints/robot_state_mae_best_absolute.pth" \
#     --data_representation absolute \
#     --evaluation_representation delta \
#     --dataset_paths "/home/najo/NAS/VLA/dataset/New_dataset6/Yellow_point" \
#     --output_dir "evaluation_results/mae_reconstruction_absolute_delta_metric" \
#     --window_size 100 \
#     --model_dim 512 \
#     --num_heads 8 \
#     --num_layers 6 \
#     --output_dim 1024

# echo ""
# # --- Evaluate 'delta' model ---
# echo "--- Evaluating 'delta' representation model (delta-space metric) ---"
# python evaluate_robot_state_mae.py \
#     --checkpoint_path "./checkpoints/robot_state_mae_best_delta.pth" \
#     --data_representation delta \
#     --evaluation_representation delta \
#     --dataset_paths "/home/najo/NAS/VLA/dataset/New_dataset6/Yellow_point" \
#     --output_dir "evaluation_results/mae_reconstruction_delta" \
#     --window_size 100 \
#     --model_dim 512 \
#     --num_heads 8 \
#     --num_layers 6 \
#     --output_dim 1024

# echo "--- Evaluating 'delta' representation model (absolute-space metric) ---"
# python evaluate_robot_state_mae.py \
#     --checkpoint_path "./checkpoints/robot_state_mae_best_delta.pth" \
#     --data_representation delta \
#     --evaluation_representation absolute \
#     --dataset_paths "/home/najo/NAS/VLA/dataset/New_dataset6/Yellow_point" \
#     --output_dir "evaluation_results/mae_reconstruction_delta_absolute_metric" \
#     --window_size 100 \
#     --model_dim 512 \
#     --num_heads 8 \
#     --num_layers 6 \
#     --output_dim 1024

echo "=============== MAE Evaluation Complete ==============="
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

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
#     --nproc_per_node=$NUM_GPUS \
#     --master_port=29501 \
#     TRAIN_SensorImage_CLIP.py \
#     --new_dataset_paths "${DATASET_PATHS_ALL[@]}" \
#     --cache_root "$CACHE_ROOT" \
#     --checkpoint_dir "$CHECKPOINT_DIR" \
#     --epochs 25 \
#     --batch_size 64 \
#     --num_workers 8 \
#     --learning_rate 2e-4 \
#     --embedding_dim 512 \
#     --gate_loss_weight 0.5 \
#     --contact_threshold 0.85 \
#     --resume_from "/home/najo/NAS/VLA/Insertion_VLAv3/checkpoints/sensor_clip_latest.pth" \
#     --sensor_noise_std 0.02

echo "=============== Sensor Encoder Pre-training Complete ==============="
echo ""

# python evaluate_sensor_representation.py \
#     --new_dataset_paths /home/najo/NAS/VLA/dataset/New_dataset4/Eye_trocar \
#     --sensor_checkpoint checkpoints/sensor_clip_best.pth \
#     --cache_root /home/najo/NAS/VLA/dataset/cache/clip_vlm_features \
#     --output_dir evaluation_results/sensor_repr_run1 \
#     --num_samples 2000 \
#     --batch_size 64 \
#     --tsne \
#     --contact_threshold 0.85 \
#     --sensor_output_dim 512


# =============================================================================
# STEP 2.5: Clean Invalid VL Cache (if exists)
# =============================================================================
# echo ""
# echo "=============== STEP 2.5: Cleaning Invalid VL Cache ==============="
# echo "Checking for and removing invalid cache files (empty image features)..."
# echo ""

# # Check if cache directory exists
# if [ -d "$CACHE_ROOT/qwen_vl_features" ]; then
#     # First do a dry run to see what would be deleted
#     python clean_invalid_cache.py --cache_root "$CACHE_ROOT/qwen_vl_features"

#     # Ask user for confirmation (optional - remove if you want automatic deletion)
#     echo ""
#     echo "❓ Do you want to delete these invalid cache files? (y/n)"
#     read -r response
#     if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
#         python clean_invalid_cache.py --cache_root "$CACHE_ROOT/qwen_vl_features" --delete
#         echo "✅ Invalid cache files deleted"
#     else
#         echo "⏭️  Skipping cache deletion"
#     fi
# else
#     echo "ℹ️  No existing cache found. Starting fresh."
# fi

# echo "=============== Cache Cleaning Complete ==============="
# echo ""


# =============================================================================
# STEP 3: Generate Action Decoder VL Cache (REQUIRED for fast training!)
# =============================================================================
echo ""
echo "=============== STEP 3: Generating Action Decoder VL Cache ==============="
echo "This will create VL feature cache for 100% of the dataset."
echo "Using vlm_reuse_count=1 for complete coverage."
echo "This will take 30-60 minutes but makes training MUCH faster!"
echo ""

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
#     --nproc_per_node=$NUM_GPUS \
#     --master_port=29502 \
#     TRAIN_FlowMatching.py \
#     --mode cache \
#     --dataset_paths "${DATASET_PATHS_ALL[@]}" \
#     --batch_size 64 \
#     --num_workers 8 \
#     --vlm_reuse_count 1 \
#     --image_resize_height 360 \
#     --image_resize_width 640 \
#     --cache_root "$CACHE_ROOT/qwen_vl_features" \
#     --skip_dataset_stats

echo "=============== Action Decoder VL Cache Generation Complete ==============="
echo ""

# =============================================================================
# STEP 4: Main VLA Training (Flow Matching)
# =============================================================================
echo ""
echo "=============== STEP 4: Main VLA Flow Matching Training ==============="
echo "Loading pre-trained sensor and robot state encoders." 
echo "Using DeepSpeed ZeRO-2."
echo "With cache enabled, training will be MUCH faster!"
echo ""

# Note: This example trains on a single dataset for simplicity.
# For multi-dataset training, add more paths to --dataset_paths.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=$NUM_GPUS \
    TRAIN_FlowMatching.py \
    --deepspeed_config configs/deepspeed_zero2_offload.json \
    --dataset_paths "${DATASET_PATHS_ALL[@]}" \
    --epochs 50 \
    --batch_size 64 \
    --grad_accum 1 \
    --lr 1e-4 \
    --min_lr 1e-7 \
    --image_resize_height 360 \
    --image_resize_width 640 \
    --sensor_enabled \
    --num_workers 8 \
    --fusion_strategy "cross_attention" \
    --finetune_vl none \
    --val_split 0.05 \
    --load_robot_state_encoder_checkpoint "$CHECKPOINT_DIR/robot_state_mae_best_delta.pth" \
    --load_sensor_encoder_checkpoint "$CHECKPOINT_DIR/sensor_clip_best.pth" \
    --action_expert_hidden_dim 1024 \
    --vlm_reuse_count 1 \
    --cache_root "$CACHE_ROOT/qwen_vl_features" \
    --use_cache --freeze_sensor_encoder --skip_dataset_stats 
    # --filter_by_cache --debug_mode
    # --resume "/home/najo/NAS/VLA/Insertion_VLAv3/checkpoints/flow_matching_best.pt" \

echo "=============== VLA Training Complete ==============="
echo ""

echo "✅✅✅ Full Training Pipeline Finished Successfully ✅✅✅"
