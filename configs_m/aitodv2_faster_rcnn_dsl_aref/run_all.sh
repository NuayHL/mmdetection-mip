#!/bin/bash

# ============================================================
# AITOD-v2  Faster R-CNN  —  DSL-AreaRefine ablation sweep
#
# DynamicSoftLabelAssignerAreaRefine: per-GT area calibration
# of the soft-label IoU ceiling for small objects.
#
# Run from repo root:
#   bash configs_m/aitodv2_faster_rcnn_dsl_aref/run_all.sh
#
# Results from aitodv2_faster_rcnn/ (baseline + USAA default)
# serve as reference — they are NOT re-run here.
# ============================================================
set -euo pipefail

EXP_PREFIX="aitodv2_faster_rcnn_dsl_aref"

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
# Phase 1 — Default DSL-AreaRefine (main result)
#
run_one "dsl_aref_default"    "configs_m/aitodv2_faster_rcnn_dsl_aref/dsl_aref_default.py"

#
# Phase 2 — calibrate_mode ablation
#   add_1 (default) vs pow — which calibration function works better?
#
run_one "dsl_aref_pow"        "configs_m/aitodv2_faster_rcnn_dsl_aref/dsl_aref_pow.py"

#
# Phase 3 — r_ref sensitivity (add_1 mode)
#   16 → boost concentrated on tiny objects (≤ 16×16)
#   32 → balanced transition (default)
#   64 → boost spread to medium objects too
#
run_one "dsl_aref_rref_16"    "configs_m/aitodv2_faster_rcnn_dsl_aref/dsl_aref_rref_16.py"
run_one "dsl_aref_rref_64"    "configs_m/aitodv2_faster_rcnn_dsl_aref/dsl_aref_rref_64.py"

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
echo ""
echo "Compare against previous baselines:"
echo "  baseline_iou      → work_dirs/aitodv2_faster_rcnn/baseline_iou/"
echo "  usaa_default      → work_dirs/aitodv2_faster_rcnn/usaa_default/"
