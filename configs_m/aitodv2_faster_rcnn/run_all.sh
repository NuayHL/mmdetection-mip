#!/bin/bash

# ============================================================
# AITOD-v2  Faster R-CNN  —  full experiment sweep
#
# Run from repo root:
#   bash configs_m/aitodv2_faster_rcnn/run_all.sh
# ============================================================
set -euo pipefail

EXP_PREFIX="aitodv2_faster_rcnn"

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
# Phase 1 — Baseline + default method
#   (run these first; if either fails, tail the log to diagnose
#    before sinking time into ablations)
#
run_one "baseline_iou"           "configs_m/aitodv2_faster_rcnn/aitodv2_iou.py"
run_one "usaa_default"           "configs_m/aitodv2_faster_rcnn/aitodv2_usaa.py"

#
# Phase 2 — Assigner variants
#
run_one "usaa_simota"            "configs_m/aitodv2_faster_rcnn/aitodv2_usaa_simota.py"
run_one "usaa_taskaligned"       "configs_m/aitodv2_faster_rcnn/aitodv2_usaa_taskaligned.py"

#
# Phase 3 — Hyperparameter ablations
#
# DynamicSoftLabelAssigner.topk
run_one "usaa_topk_5"            "configs_m/aitodv2_faster_rcnn/aitodv2_usaa_topk_5.py"
run_one "usaa_topk_20"           "configs_m/aitodv2_faster_rcnn/aitodv2_usaa_topk_20.py"

# DynamicSoftLabelAssigner.soft_center_radius
run_one "usaa_cr_1"              "configs_m/aitodv2_faster_rcnn/aitodv2_usaa_cr_1.py"
run_one "usaa_cr_5"              "configs_m/aitodv2_faster_rcnn/aitodv2_usaa_cr_5.py"

# DynamicSoftLabelAssigner.iou_weight
run_one "usaa_iw_1"              "configs_m/aitodv2_faster_rcnn/aitodv2_usaa_iw_1.py"
run_one "usaa_iw_5"              "configs_m/aitodv2_faster_rcnn/aitodv2_usaa_iw_5.py"

# QualityFocalLoss.beta
run_one "usaa_beta_1"            "configs_m/aitodv2_faster_rcnn/aitodv2_usaa_beta_1.py"
run_one "usaa_beta_3"            "configs_m/aitodv2_faster_rcnn/aitodv2_usaa_beta_3.py"

#
# Report
#
echo "===================="
echo "SWEEP FINISHED"
echo "===================="
echo "Total experiments : 12"
echo "Failed            : ${#FAILED[@]}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failures:"
    for f in "${FAILED[@]}"; do
        echo "  - ${f}  →  ${LOG_DIR}/${f}.log"
    done
fi
echo "Log directory     : ${LOG_DIR}"
echo "Work directories  : work_dirs/${EXP_PREFIX}/"
