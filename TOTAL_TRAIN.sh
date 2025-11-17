#!/bin/bash
set -e

# =============================================================================
# VL 캐시 문제 해결 기록 (2025-01-12)
# =============================================================================
#
# 문제 상황:
#   - cache_only_mode에서 "VL 캐시가 필수입니다" 에러 발생
#   - 캐시가 일부만 존재하여 학습 실패
#
# 원인 분석:
#   1. 캐시 파일 경로 구조: {cache_root}/{prompt_hash}/{episode_name}_vlm{idx}.pt
#      - prompt_hash: instruction 텍스트를 SHA256 해시화한 값 (첫 8자)
#      - 각 태스크별로 다른 instruction → 다른 prompt_hash 생성
#
#   2. 실제 캐시 상황 (기존):
#      - 캐시 디렉토리: /cache/92943a2d/ (25,969개 파일)
#      - New_dataset2: Red_point만 캐시 있음 (10/51 에피소드)
#      - New_dataset3: Red_point만 캐시 있음 (50/50 에피소드)
#      - 다른 태스크(Blue/Green/White/Yellow): 캐시 없음
#
#   3. 에러 발생 이유:
#      - cache_only_mode = True (VLM 모델 로드 안 함, 메모리 절약)
#      - 캐시 없는 샘플 → VLM 실시간 생성 시도 → VLM 없음 → 에러!
#
# 해결 방법:
#   1. 모든 태스크에 대해 VL 캐시 재생성 (현재 설정)
#   2. vlm_reuse_count=3으로 통일 (캐시 생성 & 학습)
#   3. 새로운 prompt_hash로 캐시 생성 (태스크별 instruction)
#
# 실행 순서:
#   Step 0: VL 캐시 생성 (필수) - 모든 태스크, vlm_reuse_count=3
#   Step 1: Regression 학습 (선택)
#   Step 2: Flow Matching 학습 (메인)
#
# =============================================================================

# 트레젝토리 시각화
# python3 visualize_trajectory.py \
#     "/home/najo/NAS/VLA/dataset/New_dataset2/Red_point/data_collection_20251108_061254" \
#     --sample_idx 10 --save_dir ./trajectory_plots

# NUM_GPUS=4

# ROBOT_PRETRAIN_EPOCHS=200
# TEXT_PREVIEW_COUNT=100
# VAL_SPLIT=0.05

# DATASET_PATHS=(
#     "/home/najo/NAS/VLA/dataset/New_dataset2"
#     "/home/najo/NAS/VLA/dataset/New_dataset3"
# )

# CACHE_ROOT="/home/najo/NAS/VLA/dataset/cache"

# CLIP training parameters

# echo ""
# echo "=============== 1.0 Convert Robot States to NPZ (Required for MAE Pre-training) ==============="
# echo "Converting robot_states.csv to .npz for faster loading..."
# python convert_robot_states_to_npz.py "${DATASET_PATHS}"
# echo "=============== ROBOT STATE CONVERSION COMPLETE ==============="
# echo ""

# echo ""
# echo "=============== 1.1 ROBOT STATE ENCODER PRE-TRAINING (MAE) ==============="
# echo "Epochs: $ROBOT_PRETRAIN_EPOCHS, Batch Size: 64, Window Size: 100, Mask Ratio: 0.5"
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
#     --nproc_per_node=4 \
#     --master_port=29510 \
#     TRAIN_RobotState_MAE.py \
#     --epochs 200 \
#     --batch_size 32 \
#     --learning_rate 1e-4 \
#     --weight_decay 0.01 \
#     --dataset_paths "/home/najo/NAS/VLA/dataset/New_dataset3" "/home/najo/NAS/VLA/dataset/New_dataset2" "/home/najo/NAS/VLA/dataset/New_dataset4"\
#     --val_split 0.1 \
#     --window_size 100 \
#     --mask_ratio 0.75 \
#     --model_dim 512 \
#     --num_heads 8 \
#     --num_layers 4 \
#     --output_dim 1024 \
#     --min_lr 1e-6 \
#     --warmup_ratio 0.03 \
#     --hold_ratio 0.02 \
#     --sched_on step \
#     --grad_accum 1 \
#     --num_workers 4 \
#     --joint_weight 1.0 \
#     --pose_weight 1.0 \
#     --checkpoint_dir ./checkpoints \
#     --resume_from "./checkpoints/robot_state_mae_best.pth"

# echo "=============== ROBOT STATE ENCODER PRE-TRAINING COMPLETE ==============="
# echo ""

# VLM 결과 확인해보기 (Optional Preview)
# python preview_clip_vlm_responses.py \
#     --episode_dir /home/najo/NAS/VLA/dataset/New_dataset3/Red_point/data_collection_20251110_070236 \
#     --output_dir ./clip_vlm_preview \
#     --num_samples 10 \
#     --vlm_model Qwen/Qwen2.5-VL-3B-Instruct

# ========================================================================
# NOTE: CLIP VLM Cache Building (Required before training)
# ========================================================================
# IMPORTANT: This step is REQUIRED after changing CLIP_PROMPT_TEXT or adding new datasets
# - Generates task-specific CLIP VLM caches (each task has its own prompt hash)
# - Supports multi-view images (View5 + View4)
# - Automatically skips existing caches (only generates missing ones)
# - Expected time: 1-2 hours for ~100 episodes (first run), much faster for incremental updates
# ========================================================================

# echo ""
# echo "=============== 1.0 CLIP VLM CACHE BUILDING ==============="
# echo "Building CLIP VLM feature cache for new datasets..."
# echo "Strategy: Skip existing caches, only generate missing ones"
# echo ""

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
#     --nproc_per_node=4 \
#     --master_port=29500 \
#     cache_clip_vlm_features.py \
#     --new_dataset_paths "/home/najo/NAS/VLA/dataset/New_dataset4/Eye_trocar"\
#     --cache_root "/home/najo/NAS/VLA/dataset/cache" \
#     --vlm_model "Qwen/Qwen2.5-VL-7B-Instruct" \
#     --batch_size 16 \
#     --num_workers 8

# echo ""
# echo "=============== CLIP VLM CACHE BUILDING COMPLETE ==============="
# echo ""

# echo ""
# echo "=============== 1.1 SENSOR ENCODER PRE-TRAINING (CLIP) ==============="
# echo "Epochs: $SENSOR_PRETRAIN_EPOCHS, Batch Size: $PRETRAIN_BATCH_SIZE, Sensor Window: 65"
# echo "VLM Model (for cache building if needed): Qwen/Qwen2.5-VL-7B-Instruct"
# echo "Cache Root: $CACHE_ROOT/clip_vlm_features"
# echo ""
# echo "NOTE: CLIP VLM cache must already exist (run cache_clip_vlm_features.py separately if needed)"
# echo ""

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
#     --nproc_per_node=4 \
#     --master_port=29501 \
#     TRAIN_SensorImage_CLIP.py \
#     --epochs 50 \
#     --batch_size 8 \
#     --learning_rate 1e-4 \
#     --new_dataset_paths "/home/najo/NAS/VLA/dataset/New_dataset3/*_point" "/home/najo/NAS/VLA/dataset/New_dataset2/*_point" "/home/najo/NAS/VLA/dataset/New_dataset4/Eye_trocar"\
#     --val_split 0.05 \
#     --vlm_model "Qwen/Qwen2.5-VL-7B-Instruct" \
#     --sensor_window_size 65 \
#     --sensor_output_dim 1024 \
#     --embedding_dim 512 \
#     --intermediate_vlm_dim 1536 \
#     --min_lr 1e-6 \
#     --grad_accum 2 \
#     --num_workers 8 \
#     --cache_root "/home/najo/NAS/VLA/dataset/cache" \
#     --checkpoint_dir ./checkpoints \
#     --find_unused_parameters \
#     --cache_only_mode

# echo ""
# echo "=============== SENSOR ENCODER PRE-TRAINING COMPLETE ==============="
# echo ""
    # --skip_dataset_stats \
    # --skip_cache_verification \

# TOTAL_MAIN_EPOCHS=100
# STAGE1_RATIO=0.5 # 90% of training with cache

# # Batch sizes
# PRETRAIN_BATCH_SIZE=16
# GRAD_ACCUM=8
# # Calculate epochs for each stage
# STAGE1_EPOCHS=$(printf "%.0f" $(echo "$TOTAL_MAIN_EPOCHS * $STAGE1_RATIO" | bc))
# STAGE2_EPOCHS=$(($TOTAL_MAIN_EPOCHS - $STAGE1_EPOCHS))

# # Checkpoint paths for resuming


# # Fixed training parameters
# LR=1e-4
# WEIGHT_DECAY=0.01


MAIN_BATCH_SIZE=8
CACHE_ROOT="/home/najo/NAS/VLA/dataset/cache"
QWEN_CACHE_ROOT="$CACHE_ROOT/qwen_vl_features"

## VL 답변 확인용
# python preview_vlm_responses.py \
#     --episode_dir /home/najo/NAS/VLA/dataset/New_dataset3/Red_point/data_collection_20251110_065907 \
#     --num_samples 100 \
#     --vlm_model Qwen/Qwen2.5-VL-3B-Instruct \
#     --output_dir ./vlm_preview \
#     --vlm_reuse_count 1

# =================================================================
# 3.0 VL CACHE BUILDING (REQUIRED FOR CACHE MODE)
# =================================================================
# echo ""
# echo "=============== 0. VL CACHE BUILDING ==============="
# echo "Building VL feature cache for faster training..."
#
# echo "🔍 캐시 생성 설정:"
# echo "   - 데이터셋: New_dataset2, New_dataset3 (모든 태스크)"
# echo "   - vlm_reuse_count: 3"
# echo "   - 예상 소요 시간: 30분~1시간"
# echo ""
#
# # VL Cache building doesn't use DeepSpeed (not a training step)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    --master_port=29502 \
    TRAIN_FlowMatching.py \
    --mode cache \
    --dataset_paths "/home/najo/NAS/VLA/dataset/New_dataset3/*_point" "/home/najo/NAS/VLA/dataset/New_dataset4/Eye_trocar"\
    --batch_size 16 \
    --num_workers 8 \
    --image_resize_height 360 \
    --image_resize_width 640 \
    --cache_loader_only \
    --vlm_reuse_count 1 \
    --cache_root "/home/najo/NAS/VLA/dataset/cache/qwen_vl_features"
#
# echo "=============== VL CACHE BUILDING COMPLETE ==============="
# echo ""

# =================================================================
# 3.1 MAIN VLA TRAINING (REGRESSION)
# =================================================================

IMG_HEIGHT=360
IMG_WIDTH=640

# FM_CHECKPOINT="./checkpoints/flow_matching_latest2.pt"
# REG_CHECKPOINT="./checkpoints/regression_best2.pt"

# # --- 3.1 Regression Training: Stage 1 (Cache Mode) ---
# echo ""
# echo "=============== 2.1 REGRESSION TRAINING (STAGE 1: CACHE) ==============="
# echo "Epochs: $STAGE1_EPOCHS, Batch Size: $MAIN_BATCH_SIZE, Grad Accum: $GRAD_ACCUM"

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
#     --nproc_per_node=4 \
#     --master_port=29503 \
#     TRAIN_Regression.py \
#     --dataset_paths "/home/najo/NAS/VLA/dataset/New_dataset3" \
#     --epochs 50 \
#     --batch_size 32 \
#     --grad_accum 2 \
#     --lr 1e-4 \
#     --min_lr 1e-6 \
#     --image_resize_height 360 \
#     --image_resize_width 640 \
#     --sensor_enabled \
#     --cache_root "/home/najo/NAS/VLA/dataset/cache/qwen_vl_features" \
#     --num_workers 4 \
#     --fusion_strategy "cross_attention" \
#     --sensor_enabled \
#     --finetune_vl none \
#     --val_split 0.05 \
#     --load_sensor_encoder_checkpoint "./checkpoints/sensor_clip_latest.pth" \
#     --load_robot_state_encoder_checkpoint "./checkpoints/robot_state_mae_best.pth" \
#     --vlm_reuse_count 1 \
#     --cache_root "/home/najo/NAS/VLA/dataset/cache/qwen_vl_features" \
#     --skip_dataset_stats
    # --resume $REG_CHECKPOINT

# echo "=============== REGRESSION STAGE 1 COMPLETE ==============="
# echo ""

# # --- 3.2 Regression Training: Stage 2 (Live Mode) ---
# echo ""
# echo "=============== 2.2 REGRESSION TRAINING (STAGE 2: LIVE) ==============="
# echo "Epochs: $STAGE2_EPOCHS, Batch Size: $MAIN_BATCH_SIZE, Grad Accum: $GRAD_ACCUM"
# torchrun --nproc_per_node=$NUM_GPUS TRAIN_Regression.py \
#     --epochs $STAGE2_EPOCHS \
#     --batch_size $MAIN_BATCH_SIZE \
#     --grad_accum $GRAD_ACCUM \
#     --lr $LR \
#     --dataset_paths "${DATASET_PATHS[@]}" \
#     --dataset_weights "${DATASET_WEIGHTS[@]}" \
#     --image_resize_height $IMG_HEIGHT \
#     --image_resize_width $IMG_WIDTH \
#     --fusion_strategy $FUSION \
#     $SENSOR_ENABLED \
#     $FINETUNE_ARGS \
#     --use_cache \
#     --val_split $VAL_SPLIT \
#     --cache_root $QWEN_CACHE_ROOT \
#     --resume $REG_CHECKPOINT
# echo "=============== REGRESSION STAGE 2 COMPLETE ==============="
# echo ""


# =================================================================
# 3.2 MAIN VLA TRAINING (FLOW MATCHING)
# =================================================================

# --- 3.2 Flow Matching Training: Stage 1 (Cache Mode) ---
# echo ""
# echo "=============== 3.1 FLOW MATCHING TRAINING (STAGE 1: DeepSpeed ZeRO-2) ==============="
# echo "Epochs: 50, Batch Size: 32, Grad Accum: 2"
# echo "DeepSpeed: ZeRO-2 (optimizer + gradient partitioning)"
# echo "Sensor Encoder: Lightweight (58M) - Conv 256 → 2048, Transformer 1024"
# echo ""
# "/home/najo/NAS/VLA/dataset/New_dataset2" "/home/najo/NAS/VLA/dataset/New_dataset3"
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 \
    TRAIN_FlowMatching.py \
    --deepspeed_config configs/deepspeed_zero2.json \
    --dataset_paths "/home/najo/NAS/VLA/dataset/New_dataset4/Eye_trocar" \
    --epochs 50 \
    --batch_size 32 \
    --grad_accum 2 \
    --lr 1e-4 \
    --min_lr 1e-6 \
    --image_resize_height 360 \
    --image_resize_width 640 \
    --sensor_enabled \
    --sensor_hidden_dim 128 \
    --sensor_transformer_dim 256 \
    --num_workers 4 \
    --fusion_strategy "cross_attention" \
    --finetune_vl none \
    --val_split 0.05 \
    --freeze_encoders \
    --load_robot_state_encoder_checkpoint "./checkpoints/robot_state_mae_best.pth" \
    --load_sensor_encoder_checkpoint "./checkpoints/sensor_clip_latest.pth" \
    --vlm_reuse_count 1 --cache_root /home/najo/NAS/VLA/dataset/cache/qwen_vl_features --skip_dataset_stats --use_cache --filter_by_cache

# echo "=============== FLOW MATCHING STAGE 1 COMPLETE ==============="
# echo ""
#
# echo ""
# echo "=============== 3.2 FLOW MATCHING TRAINING (STAGE 2: Full Fine-tuning) ==============="
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 \
      TRAIN_FlowMatching.py \
      --deepspeed_config configs/deepspeed_zero2.json \
      --dataset_paths "/home/najo/NAS/VLA/dataset/New_dataset4/Eye_trocar" \
      --epochs 50 \
      --batch_size 32 \
      --grad_accum 2 \
      --lr 1e-4 \
      --min_lr 1e-6 \
      --image_resize_height 360 \
      --image_resize_width 640 \
      --sensor_enabled \
      --sensor_hidden_dim 128 \
      --sensor_transformer_dim 256 \
      --num_workers 4 \
      --fusion_strategy cross_attention \
      --finetune_vl none \
      --val_split 0.05 \
      --load_robot_state_encoder_checkpoint ./checkpoints/robot_state_mae_best.pth \
      --load_sensor_encoder_checkpoint ./checkpoints/sensor_clip_latest.pth \
      --vlm_reuse_count 1 --cache_root /home/najo/NAS/VLA/dataset/cache/qwen_vl_features \
      --skip_dataset_stats --use_cache --filter_by_cache

# echo "Epochs: 10, Batch Size: 4, Grad Accum: 8"
# echo "DeepSpeed: ZeRO-2 with Full Fine-tuning (ALL models trainable)"
# echo ""
# echo "📊 Sensor Encoder: Lightweight (58M)"
# echo "   Conv: 1025 → 256 → 512 → 1024 → 2048"
# echo "   Projection: 2048 → 1024"
# echo "   Transformer: d_model=1024, 2 layers"
# echo "   Full Model: 3.88B (VLM 3.75B + Action 70M + Sensor 58M + Robot 3M)"
# echo ""
#

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 \
    TRAIN_FlowMatching.py \
    --deepspeed_config configs/deepspeed_zero2_offload.json \
    --dataset_paths "/home/najo/NAS/VLA/dataset/New_dataset6/*_point" \
    --batch_size 32 \
    --grad_accum 1 \
    --image_resize_height 360 \
    --image_resize_width 640 \
    --epochs 50 \
    --lr 1e-4 \
    --min_lr 1e-7 \
    --finetune_vl none \
    --sensor_enabled \
    --sensor_hidden_dim 128 \
    --sensor_transformer_dim 256 \
    --vlm_reuse_count 1 \
    --fusion_strategy "cross_attention" \
    --num_workers 16 \
    --skip_dataset_stats \
    --resume "./checkpoints/backup/flow_matching_best.pt"

#    --load_robot_state_encoder_checkpoint "./checkpoints/robot_state_mae_best.pth" \
    # --load_sensor_encoder_checkpoint ./checkpoints/sensor_clip_latest.pth \
# echo ""
# echo "=============== 3.3 FLOW MATCHING TRAINING (STAGE 3: Continued Training) ==============="
# echo "Epochs: 50, Batch Size: 32, Grad Accum: 2"
# echo "DeepSpeed: ZeRO-2"
# echo "Sensor Encoder: Lightweight (58M) - Conv 256 → 2048, Transformer 1024"
# echo ""
#
# CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 \
#     TRAIN_FlowMatching.py \
#     --deepspeed_config configs/deepspeed_zero2.json \
#     --dataset_paths "/home/najo/NAS/VLA/dataset/New_dataset3" \
#     --epochs 50 \
#     --batch_size 32 \
#     --grad_accum 2 \
#     --lr 1e-4 \
#     --min_lr 1e-6 \
#     --image_resize_height 360 \
#     --image_resize_width 640 \
#     --sensor_enabled \
#     --sensor_hidden_dim 256 \
#     --sensor_transformer_dim 1024 \
#     --num_workers 4 \
#     --finetune_vl full \
#     --fusion_strategy "cross_attention" \
#     --val_split 0.05 \
#     --load_robot_state_encoder_checkpoint "./checkpoints/robot_state_mae_best.pth" \
#     --vlm_reuse_count 1 \
#     --skip_dataset_stats \
#     --resume "./checkpoints/flow_matching_latest"

# --use_cache
# --cache_root "/home/najo/NAS/VLA/dataset/cache/qwen_vl_features" \

# --- 3.2 Flow Matching Training: Stage 2 (Live Mode) ---
# echo ""
# echo "=============== 3.2 FLOW MATCHING TRAINING (STAGE 2: LIVE) ==============="
# echo "Epochs: $STAGE2_EPOCHS, Batch Size: $MAIN_BATCH_SIZE, Grad Accum: $GRAD_ACCUM"
# torchrun --nproc_per_node=$NUM_GPUS TRAIN_FlowMatching.py \
#     --epochs $STAGE2_EPOCHS \
#     --batch_size $MAIN_BATCH_SIZE \
#     --grad_accum $GRAD_ACCUM \
#     --lr $LR \
#     --dataset_paths "${DATASET_PATHS[@]}" \
#     --dataset_weights "${DATASET_WEIGHTS[@]}" \
#     --image_resize_height $IMG_HEIGHT \
#     --image_resize_width $IMG_WIDTH \
#     --fusion_strategy $FUSION \
#     $SENSOR_ENABLED \
#     $FINETUNE_ARGS \
#     --val_split $VAL_SPLIT \
#     --cache_root $QWEN_CACHE_ROOT \
#     --resume $FM_CHECKPOINT
# echo "=============== FLOW MATCHING STAGE 2 COMPLETE ==============="
# echo ""

CUDA_VISIBLE_DEVICES=3 python3 evaluate_flowmatching_episode.py \
    --episode-path /home/najo/NAS/VLA/dataset/New_dataset4/Eye_trocar/data_collection_20251115_002703 \
    --checkpoint /home/najo/NAS/VLA/Insertion_VLAv3/checkpoints/backup/flow_matching_best.pt \
    --output evaluation_results/data_collection_eval.json \
    --sensor-hidden-dim 128 \
    --sensor-transformer-dim 256 \
    --batch-size 4 \
    --num-workers 0

python evaluation_results/plot_first_delta_action.py \
    --results-json /home/najo/NAS/VLA/Insertion_VLAv3/evaluation_results/data_collection_eval.json

# echo "✅✅✅ VLA FULL TRAINING PIPELINE FINISHED ✅✅✅"

# =================================================================
# 4. MODEL TEST 
# =================================================================
#
# python benchmark_realtime_inference.py \
#     --checkpoint-regression checkpoints/regression_best.pt \
#     --checkpoint-flow checkpoints/flow_matching_best.pt \
#     --dataset-dir /home/najo/NAS/VLA/dataset/New_dataset/Blue_point/episode_20251030_025856 \
#     --cache-root ./cache/qwen_vl_features \
#     --device cuda:0 \
#     --num-iterations 20 \
#     --compare-views \
#     --parallel-view-encoding

# # 센서 인코더 UAMP 등 성능 분석 시각화

# python analyze_sensor_embeddings.py \
#     --sensor-checkpoint checkpoints/sensor_clip_best.pth \
#     --dataset-paths \
#     /home/najo/NAS/VLA/dataset/New_dataset2/Green_point/data_collection_20251108_054442 \
#     --output-dir analysis/sensor_tsne \
#     --max-samples-per-episode 200 \
#     --method both \
#     --device cuda:0

# # 로봇 엔코더 성능 분석

# python analyze_robot_reproducibility.py \
#     --task-dirs /home/najo/NAS/VLA/dataset/New_dataset2/Yellow_point \
#     --exclude-episodes /home/najo/NAS/VLA/dataset/New_dataset2/Yellow_point/data_collection_20251108_043306 \
#     --output-dir analysis/reproducibility \
#     --target-length 200 \
#     --use-median

# python reconstruct_robot_states.py \
#        --episode-roots /home/najo/NAS/VLA/dataset/New_dataset2/Yellow_point \
#        --exclude-episodes /home/najo/NAS/VLA/dataset/New_dataset2/Yellow_point/data_collection_20251108_043306 \
#        --checkpoint checkpoints/robot_state_mae_best.pth \
#        --window-size 100 --stride 20 --mask-ratio 0.0 \
#        --output-root analysis/reconstructions \
#        --output-name robot_states_recon.npz \
#        --device cuda:0 --dtype bfloat16 --verbose

# python analyze_robot_reproducibility.py \
#        --task-dirs /home/najo/NAS/VLA/dataset/New_dataset2/Yellow_point \
#        --exclude-episodes /home/najo/NAS/VLA/dataset/New_dataset2/Yellow_point/data_collection_20251108_043306 \
#        --output-dir analysis/reproducibility \
#        --target-length 200 --use-median \
#        --recon-file-name robot_states_recon.npz \
#        --recon-key poses \
#        --recon-root /home/najo/NAS/VLA/Insertion_VLAv2/analysis/reconstructions

# =================================================================
# 4. ABLATION STUDIES (FOR REGRESSION MODEL)
# =================================================================
#
# NOTE: This section is for running ablation studies on the TRAIN_Regression.py script.
# The commands below use Stage 1 (cache mode) for faster experimentation.
# It is recommended to run each experiment separately and record the results from wandb.
#
# -----------------------------------------------------------------
# --- Experiment Group 1: Modality Ablation (All Views)
# -----------------------------------------------------------------

# --- Exp 1.1: Vision Only (All Views) ---
# echo "--- Running Ablation: Vision Only (All Views) ---"
# torchrun --nproc_per_node=$NUM_GPUS TRAIN_Regression.py \
#     --epochs $STAGE1_EPOCHS \
#     --batch_size $MAIN_BATCH_SIZE \
#     --grad_accum $GRAD_ACCUM \
#     --lr $LR \
#     --fusion_strategy "concat" \
#     --disable-sensor \
#     --disable-robot-state \
#     --use_cache

# --- Exp 1.2: Vision + Sensor Only ---
# echo "--- Running Ablation: Vision + Sensor Only ---"
# torchrun --nproc_per_node=$NUM_GPUS TRAIN_Regression.py \
#     --epochs $STAGE1_EPOCHS \
#     --batch_size $MAIN_BATCH_SIZE \
#     --grad_accum $GRAD_ACCUM \
#     --lr $LR \
#     --fusion_strategy "concat" \
#     --disable-robot-state \
#     --use_cache

# --- Exp 1.3: Vision + Robot State Only ---
# echo "--- Running Ablation: Vision + Robot State Only ---"
# torchrun --nproc_per_node=$NUM_GPUS TRAIN_Regression.py \
#     --epochs $STAGE1_EPOCHS \
#     --batch_size $MAIN_BATCH_SIZE \
#     --grad_accum $GRAD_ACCUM \
#     --lr $LR \
#     --fusion_strategy "concat" \
#     --disable-sensor \
#     --use_cache

# --- Exp 1.4: Full Model (All Views, All Modalities) ---
# This is the standard full run for comparison.
# echo "--- Running Ablation: Full Model (All Views) ---"
# torchrun --nproc_per_node=$NUM_GPUS TRAIN_Regression.py \
#     --epochs $STAGE1_EPOCHS \
#     --batch_size $MAIN_BATCH_SIZE \
#     --grad_accum $GRAD_ACCUM \
#     --lr $LR \
#     --fusion_strategy "concat" \
#     --use_cache

# -----------------------------------------------------------------
# --- Experiment Group 2: Per-View Importance (with Full Modalities)
# -----------------------------------------------------------------
# This loop runs a separate training for each individual view.

# for view_num in {1..5}; do
#     echo "--- Running Ablation: View $view_num Only (Full Modalities) ---"
#     torchrun --nproc_per_node=$NUM_GPUS TRAIN_Regression.py \
#         --epochs $STAGE1_EPOCHS \
#         --batch_size $MAIN_BATCH_SIZE \
#         --grad_accum $GRAD_ACCUM \
#         --lr $LR \
#         --fusion_strategy "concat" \
#         --views $view_num \
#         --use_cache
# done

# echo "✅ Ablation study section added. Uncomment the desired experiments to run."
