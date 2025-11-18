#!/bin/bash
# =============================================================================
# VLA Model Ablation Study Script
# =============================================================================
# This script runs comprehensive ablation study by evaluating the model with:
#   1. Full model (baseline)
#   2. Without Sensor Encoder
#   3. Without Robot State Encoder
#   4. Without Both Encoders (VL only)
#
# Usage:
#   bash RUN_ABLATION_STUDY.sh
# =============================================================================

set -e

# --- Configuration ---
CHECKPOINT="./checkpoints/flow_matching_best.pt"
BASE_OUTPUT_DIR="./evaluation_results/ablation_study"

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
WANDB_PROJECT="QwenVLA-Ablation"

# =============================================================================
# Helper function to run evaluation
# =============================================================================
run_evaluation() {
    local CONFIG_NAME=$1
    local EXTRA_ARGS=$2
    local OUTPUT_DIR="${BASE_OUTPUT_DIR}/${CONFIG_NAME}"
    local WANDB_NAME="ablation_${CONFIG_NAME}_$(date +%m%d_%H%M)"

    echo ""
    echo "=========================================================================="
    echo "  RUNNING: ${CONFIG_NAME}"
    echo "=========================================================================="
    echo "Output: $OUTPUT_DIR"
    echo "Extra args: $EXTRA_ARGS"
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
        --wandb-name "$WANDB_NAME" \
        $EXTRA_ARGS

    echo ""
    echo "✅ Completed: ${CONFIG_NAME}"
    echo ""
}

# =============================================================================
# Run all ablation configurations
# =============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║                  VLA MODEL ABLATION STUDY                              ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Checkpoint: $CHECKPOINT"
echo "Base output: $BASE_OUTPUT_DIR"
echo "Configurations to evaluate:"
echo "  1. full_model      - Full VLA with all components"
echo "  2. wo_sensor       - Without Sensor Encoder"
echo "  3. wo_robot_state  - Without Robot State Encoder"
echo "  4. wo_both         - Without Both Encoders (VL only)"
echo ""

# 1. Full model (baseline)
run_evaluation "full_model" ""

# 2. Without Sensor Encoder
run_evaluation "wo_sensor" "--disable-sensor"

# 3. Without Robot State Encoder
run_evaluation "wo_robot_state" "--disable-robot-state"

# 4. Without Both (VL only)
run_evaluation "wo_both" "--disable-sensor --disable-robot-state"

# =============================================================================
# Generate comparison visualizations
# =============================================================================
echo ""
echo "=========================================================================="
echo "  GENERATING ABLATION COMPARISON PLOTS"
echo "=========================================================================="
echo ""

# Find all results JSON files
FULL_JSON=$(find "${BASE_OUTPUT_DIR}/full_model" -name "evaluation_results_*.json" | head -n 1)
WO_SENSOR_JSON=$(find "${BASE_OUTPUT_DIR}/wo_sensor" -name "evaluation_results_*.json" | head -n 1)
WO_ROBOT_JSON=$(find "${BASE_OUTPUT_DIR}/wo_robot_state" -name "evaluation_results_*.json" | head -n 1)
WO_BOTH_JSON=$(find "${BASE_OUTPUT_DIR}/wo_both" -name "evaluation_results_*.json" | head -n 1)

if [ -z "$FULL_JSON" ] || [ -z "$WO_SENSOR_JSON" ] || [ -z "$WO_ROBOT_JSON" ] || [ -z "$WO_BOTH_JSON" ]; then
    echo "❌ Error: One or more evaluation results JSON not found"
    echo "   Full model: $FULL_JSON"
    echo "   w/o Sensor: $WO_SENSOR_JSON"
    echo "   w/o Robot State: $WO_ROBOT_JSON"
    echo "   w/o Both: $WO_BOTH_JSON"
    exit 1
fi

echo "Found all results JSON files:"
echo "  - Full model: $FULL_JSON"
echo "  - w/o Sensor: $WO_SENSOR_JSON"
echo "  - w/o Robot State: $WO_ROBOT_JSON"
echo "  - w/o Both: $WO_BOTH_JSON"
echo ""

python evaluation_results/plot_ablation_comparison.py \
    --full-model "$FULL_JSON" \
    --wo-sensor "$WO_SENSOR_JSON" \
    --wo-robot-state "$WO_ROBOT_JSON" \
    --wo-both "$WO_BOTH_JSON" \
    --output-dir "${BASE_OUTPUT_DIR}/comparison_plots" \
    --max-trajectory-samples 3

echo ""
echo "=========================================================================="
echo "  ABLATION STUDY COMPLETE"
echo "=========================================================================="
echo ""
echo "Results saved to: $BASE_OUTPUT_DIR"
echo ""
echo "Individual evaluation results:"
echo "  - Full model:       ${BASE_OUTPUT_DIR}/full_model/"
echo "  - w/o Sensor:       ${BASE_OUTPUT_DIR}/wo_sensor/"
echo "  - w/o Robot State:  ${BASE_OUTPUT_DIR}/wo_robot_state/"
echo "  - w/o Both:         ${BASE_OUTPUT_DIR}/wo_both/"
echo ""
echo "Comparison plots:     ${BASE_OUTPUT_DIR}/comparison_plots/"
echo ""
echo "📊 Quick view of comparison plots:"
ls -lh "${BASE_OUTPUT_DIR}/comparison_plots/"
echo ""
