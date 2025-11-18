#!/bin/bash
# =============================================================================
# VLA Model View Ablation Study Script
# =============================================================================
# This script runs comprehensive view ablation study by evaluating the model with:
#   1. All views (baseline - typically 5 views: 0,1,2,3,4)
#   2. Different number of views (4, 3, 2, 1 views)
#   3. Each individual view (view 0, 1, 2, 3, 4 separately)
#
# Also measures inference time for each configuration to analyze the trade-off
# between performance and computational cost.
#
# Usage:
#   bash RUN_VIEW_ABLATION_STUDY.sh
# =============================================================================

set -e

# --- Configuration ---
CHECKPOINT="./checkpoints/flow_matching_best.pt"
BASE_OUTPUT_DIR="./evaluation_results/view_ablation_study"

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
WANDB_PROJECT="QwenVLA-ViewAblation"

# =============================================================================
# Helper function to run evaluation
# =============================================================================
run_evaluation() {
    local CONFIG_NAME=$1
    local VIEW_ARGS=$2
    local OUTPUT_DIR="${BASE_OUTPUT_DIR}/${CONFIG_NAME}"
    local WANDB_NAME="view_ablation_${CONFIG_NAME}_$(date +%m%d_%H%M)"

    echo ""
    echo "=========================================================================="
    echo "  RUNNING: ${CONFIG_NAME}"
    echo "=========================================================================="
    echo "Output: $OUTPUT_DIR"
    echo "View args: $VIEW_ARGS"
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
        $VIEW_ARGS

    echo ""
    echo "✅ Completed: ${CONFIG_NAME}"
    echo ""
}

# =============================================================================
# Run all view ablation configurations
# =============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║                  VLA MODEL VIEW ABLATION STUDY                         ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Checkpoint: $CHECKPOINT"
echo "Base output: $BASE_OUTPUT_DIR"
echo ""
echo "Configurations to evaluate:"
echo "  1. all_views       - All views (baseline)"
echo "  2. views_0_1_2_3   - 4 views (exclude view 4)"
echo "  3. views_0_1_2     - 3 views (first 3)"
echo "  4. views_0_1       - 2 views (first 2)"
echo "  5. view_0          - Single view (view 0 only)"
echo "  6. view_1          - Single view (view 1 only)"
echo "  7. view_2          - Single view (view 2 only)"
echo "  8. view_3          - Single view (view 3 only)"
echo "  9. view_4          - Single view (view 4 only)"
echo ""

# 1. All views (baseline)
run_evaluation "all_views" ""

# 2. Different number of views
run_evaluation "views_0_1_2_3" "--view-indices 0 1 2 3"
run_evaluation "views_0_1_2" "--view-indices 0 1 2"
run_evaluation "views_0_1" "--view-indices 0 1"

# 3. Each individual view
run_evaluation "view_0" "--view-indices 0"
run_evaluation "view_1" "--view-indices 1"
run_evaluation "view_2" "--view-indices 2"
run_evaluation "view_3" "--view-indices 3"
run_evaluation "view_4" "--view-indices 4"

# =============================================================================
# Generate comparison visualizations
# =============================================================================
echo ""
echo "=========================================================================="
echo "  GENERATING VIEW ABLATION COMPARISON PLOTS"
echo "=========================================================================="
echo ""

# Find all results JSON files
ALL_VIEWS_JSON=$(find "${BASE_OUTPUT_DIR}/all_views" -name "evaluation_results_*.json" | head -n 1)
VIEWS_0123_JSON=$(find "${BASE_OUTPUT_DIR}/views_0_1_2_3" -name "evaluation_results_*.json" | head -n 1)
VIEWS_012_JSON=$(find "${BASE_OUTPUT_DIR}/views_0_1_2" -name "evaluation_results_*.json" | head -n 1)
VIEWS_01_JSON=$(find "${BASE_OUTPUT_DIR}/views_0_1" -name "evaluation_results_*.json" | head -n 1)
VIEW_0_JSON=$(find "${BASE_OUTPUT_DIR}/view_0" -name "evaluation_results_*.json" | head -n 1)
VIEW_1_JSON=$(find "${BASE_OUTPUT_DIR}/view_1" -name "evaluation_results_*.json" | head -n 1)
VIEW_2_JSON=$(find "${BASE_OUTPUT_DIR}/view_2" -name "evaluation_results_*.json" | head -n 1)
VIEW_3_JSON=$(find "${BASE_OUTPUT_DIR}/view_3" -name "evaluation_results_*.json" | head -n 1)
VIEW_4_JSON=$(find "${BASE_OUTPUT_DIR}/view_4" -name "evaluation_results_*.json" | head -n 1)

# Check if all results exist
MISSING=0
for json_file in "$ALL_VIEWS_JSON" "$VIEWS_0123_JSON" "$VIEWS_012_JSON" "$VIEWS_01_JSON" \
                 "$VIEW_0_JSON" "$VIEW_1_JSON" "$VIEW_2_JSON" "$VIEW_3_JSON" "$VIEW_4_JSON"; do
    if [ -z "$json_file" ]; then
        MISSING=1
    fi
done

if [ $MISSING -eq 1 ]; then
    echo "❌ Error: One or more evaluation results JSON not found"
    echo "   All views:     $ALL_VIEWS_JSON"
    echo "   Views 0-3:     $VIEWS_0123_JSON"
    echo "   Views 0-2:     $VIEWS_012_JSON"
    echo "   Views 0-1:     $VIEWS_01_JSON"
    echo "   View 0:        $VIEW_0_JSON"
    echo "   View 1:        $VIEW_1_JSON"
    echo "   View 2:        $VIEW_2_JSON"
    echo "   View 3:        $VIEW_3_JSON"
    echo "   View 4:        $VIEW_4_JSON"
    exit 1
fi

echo "Found all results JSON files ✅"
echo ""

python evaluation_results/plot_view_ablation_comparison.py \
    --all-views "$ALL_VIEWS_JSON" \
    --views-4 "$VIEWS_0123_JSON" \
    --views-3 "$VIEWS_012_JSON" \
    --views-2 "$VIEWS_01_JSON" \
    --view-0 "$VIEW_0_JSON" \
    --view-1 "$VIEW_1_JSON" \
    --view-2 "$VIEW_2_JSON" \
    --view-3 "$VIEW_3_JSON" \
    --view-4 "$VIEW_4_JSON" \
    --output-dir "${BASE_OUTPUT_DIR}/comparison_plots" \
    --max-trajectory-samples 3

echo ""
echo "=========================================================================="
echo "  VIEW ABLATION STUDY COMPLETE"
echo "=========================================================================="
echo ""
echo "Results saved to: $BASE_OUTPUT_DIR"
echo ""
echo "Individual evaluation results:"
echo "  - All views:      ${BASE_OUTPUT_DIR}/all_views/"
echo "  - 4 views (0-3):  ${BASE_OUTPUT_DIR}/views_0_1_2_3/"
echo "  - 3 views (0-2):  ${BASE_OUTPUT_DIR}/views_0_1_2/"
echo "  - 2 views (0-1):  ${BASE_OUTPUT_DIR}/views_0_1/"
echo "  - View 0 only:    ${BASE_OUTPUT_DIR}/view_0/"
echo "  - View 1 only:    ${BASE_OUTPUT_DIR}/view_1/"
echo "  - View 2 only:    ${BASE_OUTPUT_DIR}/view_2/"
echo "  - View 3 only:    ${BASE_OUTPUT_DIR}/view_3/"
echo "  - View 4 only:    ${BASE_OUTPUT_DIR}/view_4/"
echo ""
echo "Comparison plots:   ${BASE_OUTPUT_DIR}/comparison_plots/"
echo ""
echo "📊 Quick view of comparison plots:"
ls -lh "${BASE_OUTPUT_DIR}/comparison_plots/"
echo ""
