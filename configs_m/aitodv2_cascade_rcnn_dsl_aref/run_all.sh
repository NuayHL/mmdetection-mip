#!/bin/bash

# ============================================================
# AITOD-v2  Cascade R-CNN  —  DSL-DYAB-RCNN size-dependent
#
# Applies the DSL-DYAB-RCNN (size-dependent expansion + DyabDSL)
# improvement to the Cascade R-CNN ResNet50+FPN backbone.
#
# Uses DynAssignRoIHead (same head as the Faster R-CNN DSL version)
# to isolate backbone differences from the RoI head.
#
# Run from repo root:
#   bash configs_m/aitodv2_cascade_rcnn_dsl_aref/run_all.sh
# ============================================================
set -euo pipefail

EXP_PREFIX="aitodv2_cascade_rcnn_dsl_aref"

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
# DSL-DYAB-RCNN size-dependent expansion
#
run_one "dsl_dyab_rcnn_adapt"  "configs_m/aitodv2_cascade_rcnn_dsl_aref/dsl_dyab_rcnn_adapt.py"

#
# Report
#
echo "===================="
echo "SWEEP FINISHED"
echo "===================="
echo "Total experiments : 1"
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
echo "Compare against baselines:"
echo "  cascade_rcnn baseline  → work_dirs/aitodv2_cascade_rcnn/aitodv2_iou_cascade_rcnn_2x/"
echo "  faster_rcnn DSL adapt  → work_dirs/aitodv2_faster_rcnn_dsl_aref/dsl_dyab_rcnn_adapt/"
