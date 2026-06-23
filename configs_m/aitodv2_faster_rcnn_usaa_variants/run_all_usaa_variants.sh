#!/bin/bash

# ============================================================
# AITOD-v2  plain Faster R-CNN  —  USAA variant screening
#
# Tests every USAA two-stage implementation against the plain
# MaxIoU baseline (NO NWD-RKA / NO RFLA), to find which soft-label
# placement is strongest on the vanilla detector before moving to
# stronger two-stage backbones (Cascade R-CNN, DetectoRS).
#
# Run from repo root:
#   bash configs_m/aitodv2_faster_rcnn_usaa_variants/run_all_usaa_variants.sh
# ============================================================
set -euo pipefail

EXP_PREFIX="aitodv2_faster_rcnn_usaa_variants"

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
# Reference baseline (plain MaxIoU Faster R-CNN, mAP 0.132)
#
# run_one "baseline"        "configs_m/aitodv2_faster_rcnn/aitodv2_iou.py"

#
# USAA two-stage variants (each = baseline + exactly one soft mechanism)
#
run_one "softrpn"         "configs_m/aitodv2_faster_rcnn_usaa_variants/softrpn.py"
run_one "softlabel_roi"   "configs_m/aitodv2_faster_rcnn_usaa_variants/softlabel_roi.py"
run_one "iouaware"        "configs_m/aitodv2_faster_rcnn_usaa_variants/iouaware.py"

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
