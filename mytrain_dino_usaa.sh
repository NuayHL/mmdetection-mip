#!/bin/bash

# -----------------------------
# DINO baseline vs USAA flavour on AITOD-v2 (A/B comparison)
#   baseline : HungarianAssigner        + FocalLoss (hard label)
#   usaa     : HungarianAssignerUSAA     + QualityFocalLoss (calibrated IoU
#              soft label, DINOHeadUSAA)
# -----------------------------

set -e

# Experiment prefix
EXP_PREFIX="aitodv2_dino_usaa"

# Log directory
LOG_DIR="terminal_log/terminal_log_${EXP_PREFIX}"
mkdir -p "$LOG_DIR"

echo "===================="
echo "STARTING TRAINING"
echo "Experiment prefix: ${EXP_PREFIX}"
echo "Log directory: ${LOG_DIR}"
echo "===================="

# ---- ours: DINO + USAA (scale-aware matching + calibrated soft label) ----
EXP_NAME="dino-4scale_r50_usaa_aitodv2"
CONFIG="configs_m/dino_tiny/dino-4scale_r50_usaa_aitodv2.py"
echo "Experiment name: ${EXP_NAME}"
echo "TRAINING..."
python tools/train.py "$CONFIG" \
                      --work-dir ./work_dirs/"${EXP_PREFIX}/${EXP_NAME}" \
                      > "$LOG_DIR/${EXP_NAME}.log" 2>&1
echo "TRAINING FINISHED"
echo "Log saved to: $LOG_DIR/${EXP_NAME}.log"

# ---- baseline: vanilla DINO (uncomment to run the A/B counterpart) ----
# EXP_NAME="dino-4scale_r50_baseline_aitodv2"
# CONFIG="configs_m/dino_tiny/dino-4scale_r50_1xb2-12e_aitodv2.py"
# echo "Experiment name: ${EXP_NAME}"
# echo "TRAINING..."
# python tools/train.py "$CONFIG" \
#                       --work-dir ./work_dirs/"${EXP_PREFIX}/${EXP_NAME}" \
#                       > "$LOG_DIR/${EXP_NAME}.log" 2>&1
# echo "TRAINING FINISHED"
# echo "Log saved to: $LOG_DIR/${EXP_NAME}.log"

echo "ALL DONE."
