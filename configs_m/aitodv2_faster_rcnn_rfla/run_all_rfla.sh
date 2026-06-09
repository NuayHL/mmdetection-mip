#!/bin/bash

# ============================================================
# AITOD-v2  Faster R-CNN  RFLA — full experiment sweep
#
# Run from repo root:
#   bash configs_m/aitodv2_faster_rcnn_rfla/run_all_rfla.sh
# ============================================================
set -euo pipefail

EXP_PREFIX="aitodv2_faster_rcnn_rfla"

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
# Phase 1 — RFLA base (KLD / WD)
#
run_one "rfla_kld"         "configs_m/aitodv2_faster_rcnn_rfla/aitodv2_rfla_kld.py"
run_one "rfla_wd"          "configs_m/aitodv2_faster_rcnn_rfla/aitodv2_rfla_wd.py"

#
# Phase 2 — RFLA + DSL-DYAB (KLD / WD)
#
run_one "rfla_kld_dyab"    "configs_m/aitodv2_faster_rcnn_rfla_dsl/rfla_dsl_dyab.py"
run_one "rfla_kld_dyab_balanced"    "configs_m/aitodv2_faster_rcnn_rfla_dsl/rfla_dsl_balanced.py"

run_one "rfla_wd_dyab"     "configs_m/aitodv2_faster_rcnn_rfla_dsl/rfla_wd_dsl_dyab.py"
run_one "rfla_wd_dyab_balanced"     "configs_m/aitodv2_faster_rcnn_rfla_dsl/rfla_wd_dsl_balanced.py"

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
