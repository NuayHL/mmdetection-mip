#!/bin/bash

# ============================================================
# AITOD-v2  Faster R-CNN  NWD-RKA + decoupled IoU-aware head
#   (USAA soft-IoU, placed so it does NOT lower the cls
#    ranking score; fused as score = cls * sigmoid(iou)^alpha)
#
# Run from repo root:
#   bash configs_m/aitodv2_faster_rcnn_nwdrka_iouaware/run_all_iouaware.sh
#
# Compare every run against the NWD-RKA baseline (mAP 0.216):
#   work_dirs/aitodv2_faster_rcnn_nwdrka/nwdrka
# ============================================================
set -euo pipefail

EXP_PREFIX="aitodv2_faster_rcnn_nwdrka_iouaware"

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
# Phase 1 — faithful soft IoU label (alpha=1, area-refine add_1)
#
run_one "nwdrka_iouaware"        "configs_m/aitodv2_faster_rcnn_nwdrka_iouaware/nwdrka_iouaware.py"

#
# Phase 2 — ablations
#
run_one "nwdrka_iouaware_a05"    "configs_m/aitodv2_faster_rcnn_nwdrka_iouaware/nwdrka_iouaware_a05.py"
run_one "nwdrka_iouaware_nocal"  "configs_m/aitodv2_faster_rcnn_nwdrka_iouaware/nwdrka_iouaware_nocal.py"
run_one "nwdrka_iouaware_pow"    "configs_m/aitodv2_faster_rcnn_nwdrka_iouaware/nwdrka_iouaware_pow.py"

#
# Report
#
echo "===================="
echo "SWEEP FINISHED"
echo "===================="
echo "Total experiments : 4"
echo "Failed            : ${#FAILED[@]}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failures:"
    for f in "${FAILED[@]}"; do
        echo "  - ${f}  →  ${LOG_DIR}/${f}.log"
    done
fi
echo "Log directory     : ${LOG_DIR}"
echo "Work directories  : work_dirs/${EXP_PREFIX}/"
