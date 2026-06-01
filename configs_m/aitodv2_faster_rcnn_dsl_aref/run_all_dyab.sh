#!/bin/bash

# ============================================================
# AITOD-v2  Faster R-CNN  —  DSL-DScale-DYAB ablation sweep
#
# DynamicSoftLabelAssignerDScaleDYAB: three enhancements
#   expansion   – candidate region expanded by scale_ratio × stride
#   dyab        – dynamic α/β cost weighting
#   area-refine – soft-label IoU ceiling calibration
#
# Run from repo root:
#   bash configs_m/aitodv2_faster_rcnn_dsl_aref/run_all_dyab.sh
#
# Reference: dsl_aref_default (area-refine only, no expansion/dyab)
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
# Phase 1 — Default DSL-DScale-DYAB (main result)
#
run_one "dsl_dyab_default"  "configs_m/aitodv2_faster_rcnn_dsl_aref/dsl_dyab_default.py"

#
# Phase 2 — scale_ratio ablation
#   0.0 = strict (no expansion)   1.0 = default   2.0 = aggressive
#
run_one "dsl_dyab_func1"    "configs_m/aitodv2_faster_rcnn_dsl_aref/dsl_dyab_func1.py"
run_one "dsl_dyab_expand"   "configs_m/aitodv2_faster_rcnn_dsl_aref/dsl_dyab_expand.py"

#
# Phase 3 — dyab ablation
#   DyabCalibrationAware (default) vs DyabBudgetShift
#
run_one "dsl_dyab_budget"   "configs_m/aitodv2_faster_rcnn_dsl_aref/dsl_dyab_budget.py"

#
# Phase 4 — calibrate_mode & r_ref
#
run_one "dsl_dyab_pow"      "configs_m/aitodv2_faster_rcnn_dsl_aref/dsl_dyab_pow.py"
run_one "dsl_dyab_rref_16"  "configs_m/aitodv2_faster_rcnn_dsl_aref/dsl_dyab_rref_16.py"

#
# Report
#
echo "===================="
echo "SWEEP FINISHED"
echo "===================="
echo "Total experiments : 6"
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
echo "  dsl_aref_default → work_dirs/aitodv2_faster_rcnn_dsl_aref/dsl_aref_default/"
echo "  usaa_default     → work_dirs/aitodv2_faster_rcnn/usaa_default/"
