#!/bin/bash
# ==============================================================================
# Async Real-time VLA Inference Runner
# ==============================================================================
#
# This script runs the async real-time inference receiver for VLA model.
# It connects to camera streams, robot state, and sensor data via ZMQ/UDP.
#
# Usage:
#   ./run_async_inference.sh [--save-data] [--verbose]
#
# Options:
#   --save-data    Save images, sensor data, and robot state
#   --verbose      Enable verbose logging
#
# ==============================================================================

# Configuration
CHECKPOINT="../checkpoints/flow_matching_latest.pt"
ROBOT_IP="10.130.41.111"
MODEL_TYPE="flow_matching"
FLOW_STEPS=10
VL_REUSE=4
TASK_NAME="Red_point"

# Default start joint positions (degrees)
START_JOINTS=(191 1 309 1 92 2)

# Parse command line arguments
SAVE_DATA_FLAG=""
VERBOSE_FLAG=""
AUTO_START_FLAG="--auto-start"

while [[ $# -gt 0 ]]; do
    case $1 in
        --save-data)
            SAVE_DATA_FLAG="--save-data"
            shift
            ;;
        --verbose)
            VERBOSE_FLAG="--verbose"
            shift
            ;;
        --no-auto-start)
            AUTO_START_FLAG=""
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--save-data] [--verbose] [--no-auto-start]"
            exit 1
            ;;
    esac
done

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint not found at $CHECKPOINT"
    echo "Available checkpoints:"
    ls -lh ../checkpoints/*.pt 2>/dev/null || echo "No checkpoints found!"
    exit 1
fi

# Print configuration
echo "=================================="
echo "🚀 Async VLA Inference Runner"
echo "=================================="
echo "Checkpoint: $CHECKPOINT"
echo "Robot IP: $ROBOT_IP"
echo "Model Type: $MODEL_TYPE"
echo "Flow Steps: $FLOW_STEPS"
echo "VL Reuse: ${VL_REUSE}x"
echo "Task Name: $TASK_NAME"
echo "Start Joints: ${START_JOINTS[*]}"
echo "Auto Start: $([ -z "$AUTO_START_FLAG" ] && echo "No" || echo "Yes")"
echo "Save Data: $([ -z "$SAVE_DATA_FLAG" ] && echo "No" || echo "Yes")"
echo "Verbose: $([ -z "$VERBOSE_FLAG" ] && echo "No" || echo "Yes")"
echo "=================================="
echo ""

# Change to script directory
cd "$(dirname "$0")" || exit 1

# Run the inference receiver
python3 Async_inference_receiver.py \
    --checkpoint "$CHECKPOINT" \
    --robot-ip "$ROBOT_IP" \
    --model-type "$MODEL_TYPE" \
    --flow-steps "$FLOW_STEPS" \
    --vl-reuse "$VL_REUSE" \
    --task-name "$TASK_NAME" \
    --start-joints "${START_JOINTS[@]}" \
    $AUTO_START_FLAG \
    $SAVE_DATA_FLAG \
    $VERBOSE_FLAG

# Capture exit code
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Inference completed successfully"
else
    echo ""
    echo "❌ Inference failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
