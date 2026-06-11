#!/bin/bash

# ============================================================
# AITOD-v2  Faster R-CNN  RFLA + decoupled IoU-aware head
#   (USAA soft-IoU, placed so it does NOT lower the cls
#    ranking score; fused as score = cls * sigmoid(iou)^alpha)
#
# Run from repo root:
#   bash configs_m/aitodv2_faster_rcnn_rfla_iouaware/run_all_iouaware.sh
#
# Compare each run against the matching RFLA baseline:
#   rfla_iouaware_kld  vs  rfla_kld  (mAP 0.216)
#   rfla_iouaware_wd   vs  rfla_wd   (mAP 0.222)
# ============================================================
set -euo pipefail

EXP_PREFIX="aitodv2_faster_rcnn_rfla_iouaware"

LOG_DIR="terminal_log/terminal_log_${EXP_PREFIX}"
mkdir -p "$LOG_DIR"

FAILED=()

run_one() {
    local exp_name="$1"
    local config_path="$2"

    echo "=============================================="
    echo "  Experiment : ${exp_name}"
    echo "  Config     : ${config_path}"
    echo "  Log        : ${LOG_DIR}/${exp_name}.log"
    echo "  Started at : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================="

    if python tools/train.py "${config_path}" \
        --work-dir "./work_dirs/${EXP_PREFIX}/${exp_name}" \
        > "${LOG_DIR}/${exp_name}.log" 2>&1; then
        echo "  [PASS]  ${exp_name}  finished  $(date '+%Y-%m-%d %H:%M:%S')"
    else
        echo "  [FAIL]  ${exp_name}  failed    $(date '+%Y-%m-%d %H:%M:%S')"
        FAILED+=("${exp_name}")
    fi
    echo ""
}

#
# RFLA + IoU-aware head (KLD / WD)
#
run_one "rfla_iouaware_kld"   "configs_m/aitodv2_faster_rcnn_rfla_iouaware/rfla_iouaware_kld.py"
run_one "rfla_iouaware_wd"    "configs_m/aitodv2_faster_rcnn_rfla_iouaware/rfla_iouaware_wd.py"

#
# Report
#
echo "===================="
echo "SWEEP FINISHED"
echo "===================="
echo "Total experiments : 2"
echo "Failed            : ${#FAILED[@]}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failures:"
    for f in "${FAILED[@]}"; do
        echo "  - ${f}  →  ${LOG_DIR}/${f}.log"
    done
fi
echo "Log directory     : ${LOG_DIR}"
echo "Work directories  : work_dirs/${EXP_PREFIX}/"
