#!/bin/bash
# ============================================================
# AITOD-v2  Faster R-CNN  —  DSL-DYAB ablation sweep
#
# Experiments:
#   Phase 1 — RCNN ablaions (DyabDSL + expansion variants)
#   Phase 2 — RPN expansion (RPNExpandAssigner)
#   Phase 3 — Combined RPN + RCNN
#
# Baselines: dsl_aref_default (area-refine only, no expansion/dyab)
#            usaa_default     (original DynAssignRoIHead baseline)
#
# Run from repo root:
#   bash configs_m/aitodv2_faster_rcnn_dsl_aref/run_all_dyab.sh
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
# Phase 1 — RCNN (DyabDSL + expansion)
#
run_one "dsl_dyab_rcnn_default"  "configs_m/aitodv2_faster_rcnn_dsl_aref/dsl_dyab_rcnn_default.py"
run_one "dsl_dyab_rcnn_adapt"    "configs_m/aitodv2_faster_rcnn_dsl_aref/dsl_dyab_rcnn_adapt.py"
run_one "dsl_dyab_rcnn_noexp"    "configs_m/aitodv2_faster_rcnn_dsl_aref/dsl_dyab_rcnn_noexp.py"

#
# Phase 2 — RPN (expansion only, RCNN unchanged)
#
run_one "dsl_dyab_rpn_default"   "configs_m/aitodv2_faster_rcnn_dsl_aref/dsl_dyab_rpn_default.py"
run_one "dsl_dyab_rpn_adapt"     "configs_m/aitodv2_faster_rcnn_dsl_aref/dsl_dyab_rpn_adapt.py"

#
# Phase 3 — Combined RPN + RCNN
#
run_one "dsl_dyab_full_default"  "configs_m/aitodv2_faster_rcnn_dsl_aref/dsl_dyab_full_default.py"
run_one "dsl_dyab_full_adapt"    "configs_m/aitodv2_faster_rcnn_dsl_aref/dsl_dyab_full_adapt.py"

#
# Report
#
echo "===================="
echo "SWEEP FINISHED"
echo "===================="
echo "Total experiments : 7"
echo "Failed            : ${#FAILED[@]}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failures:"
    for f in "${FAILED[@]}"; do
        echo "  - ${f}  →  ${LOG_DIR}/${f}.log"
    done
fi
echo "Log directory     : ${LOG_DIR}"
echo ""
echo "Baselines for comparison:"
echo "  dsl_aref_default  → work_dirs/.../dsl_aref_default/"
echo "  usaa_default      → work_dirs/aitodv2_faster_rcnn/usaa_default/"
