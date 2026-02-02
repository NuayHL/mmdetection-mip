#!/bin/bash

# -----------------------------
# Script to train MMDetection model
# -----------------------------

# Experiment prefix
EXP_PREFIX="aitodv1"

# Log directory
LOG_DIR="terminal_log/terminal_log_${EXP_PREFIX}"
mkdir -p "$LOG_DIR"

# Start message
echo "===================="
echo "STARTING TRAINING"
echo "Experiment prefix: ${EXP_PREFIX}"
echo "Log directory: ${LOG_DIR}"
echo "===================="

EXP_NAME="faster-rcnn_r50_fpn_m2x_iou_aitodv1"
CONFIG="configs_m/faster_rcnn/faster-rcnn_r50_fpn_m2x_iou_aitodv1.py"
echo "Experiment name: ${EXP_NAME}"
echo "TRAINING..."
python tools/train.py "$CONFIG" \
                      --work-dir ./work_dirs/"${EXP_PREFIX}/${EXP_NAME}" \
                      > "$LOG_DIR/${EXP_NAME}.log" 2>&1
# Done message
echo "TRAINING FINISHED"
echo "Log saved to: $LOG_DIR/${EXP_NAME}.log"

EXP_NAME="faster-rcnn_r50_fpn_m2x_sim_aitodv1"
CONFIG="configs_m/faster_rcnn/faster-rcnn_r50_fpn_m2x_sim_aitodv1.py"
echo "Experiment name: ${EXP_NAME}"
echo "TRAINING..."
python tools/train.py "$CONFIG" \
                      --work-dir ./work_dirs/"${EXP_PREFIX}/${EXP_NAME}" \
                      > "$LOG_DIR/${EXP_NAME}.log" 2>&1
# Done message
echo "TRAINING FINISHED"
echo "Log saved to: $LOG_DIR/${EXP_NAME}.log"

EXP_NAME="faster-rcnn_r50_fpn_m2x_hausdorff90_aitodv1"
CONFIG="configs_m/faster_rcnn/faster-rcnn_r50_fpn_m2x_hausdorff90_aitodv1.py"
echo "Experiment name: ${EXP_NAME}"
echo "TRAINING..."
python tools/train.py "$CONFIG" \
                      --work-dir ./work_dirs/"${EXP_PREFIX}/${EXP_NAME}" \
                      > "$LOG_DIR/${EXP_NAME}.log" 2>&1
# Done message
echo "TRAINING FINISHED"
echo "Log saved to: $LOG_DIR/${EXP_NAME}.log"
