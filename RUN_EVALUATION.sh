#!/bin/bash
# =============================================================================
# VLA Model Evaluation Script
# =============================================================================
# This script demonstrates how to evaluate a trained Flow Matching VLA model
# using the comprehensive evaluation pipeline.
#
# Usage:
#   bash RUN_EVALUATION.sh
# =============================================================================

set -e

# --- Configuration ---
CHECKPOINT="./checkpoints/flow_matching_best.pt"
OUTPUT_DIR="./evaluation_results/flow_best_eval"

# Dataset paths (supports glob patterns)
DATASET_PATHS=(
    "/home/najo/NAS/VLA/dataset/New_dataset6/Red_point"
    "/home/najo/NAS/VLA/dataset/New_dataset6/Blue_point"
    "/home/najo/NAS/VLA/dataset/New_dataset6/Green_point"
)

# Evaluation parameters
BATCH_SIZE=4
NUM_WORKERS=4
THRESHOLD_MM=5.0

# Model architecture (must match training config)
SENSOR_HIDDEN_DIM=512
SENSOR_TRANSFORMER_DIM=1024

# Optional: wandb logging
WANDB_PROJECT="QwenVLA-Evaluation"
WANDB_NAME="flow_best_eval_$(date +%m%d_%H%M)"

# =============================================================================
# STEP 1: Run comprehensive evaluation
# =============================================================================
echo ""
echo "=============== STEP 1: Running Comprehensive Evaluation ==============="
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_DIR"
echo ""

python EVAL_FlowMatching.py \
    --checkpoint "$CHECKPOINT" \
    --dataset-paths "${DATASET_PATHS[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --threshold-mm $THRESHOLD_MM \
    --sensor-hidden-dim $SENSOR_HIDDEN_DIM \
    --sensor-transformer-dim $SENSOR_TRANSFORMER_DIM \
    --use-cache \
    --cache-only-mode \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-name "$WANDB_NAME"

echo "=============== Evaluation Complete ==============="
echo ""

# =============================================================================
# STEP 2: Generate visualization plots
# =============================================================================
echo ""
echo "=============== STEP 2: Generating Visualization Plots ==============="
echo ""

# Find the results JSON file
RESULTS_JSON=$(find "$OUTPUT_DIR" -name "evaluation_results_*.json" | head -n 1)

if [ -z "$RESULTS_JSON" ]; then
    echo "❌ Error: No evaluation results JSON found in $OUTPUT_DIR"
    exit 1
fi

echo "Results JSON: $RESULTS_JSON"
echo ""

python evaluation_results/plot_evaluation_results.py \
    --results-json "$RESULTS_JSON" \
    --output-dir "$OUTPUT_DIR/plots" \
    --max-trajectory-samples 5

echo "=============== Visualization Complete ==============="
echo ""

# =============================================================================
# STEP 3: Display summary
# =============================================================================
echo ""
echo "✅✅✅ Evaluation Pipeline Finished Successfully ✅✅✅"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "   - JSON results: $RESULTS_JSON"
echo "   - Plots: $OUTPUT_DIR/plots/"
echo ""
echo "📊 Quick view of plots:"
ls -lh "$OUTPUT_DIR/plots/"
echo ""
