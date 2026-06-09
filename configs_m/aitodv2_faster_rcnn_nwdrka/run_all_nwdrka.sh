#!/bin/bash

# ============================================================
# AITOD-v2  Faster R-CNN  NWD-RKA — full experiment sweep
#
# Run from repo root:
#   bash configs_m/aitodv2_faster_rcnn_nwdrka/run_all_nwdrka.sh
#
# Phase 1 trains plain NWD-RKA (RankingAssigner + NWD in the RPN).
# Phase 2 trains NWD-RKA combined with DSL-DYAB (my RCNN method).
# ============================================================
set -euo pipefail

EXP_PREFIX="aitodv2_faster_rcnn_nwdrka"

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
# Phase 2 — NWD-RKA + DSL-DYAB (my method)
#
# run_one "nwdrka_dyab"      "configs_m/aitodv2_faster_rcnn_nwdrka_dsl/nwdrka_dsl_dyab.py"

# #
# # Phase 1 — NWD-RKA base
# #
# run_one "nwdrka"           "configs_m/aitodv2_faster_rcnn_nwdrka/aitodv2_nwdrka.py"

# run_one "nwdrka_softiou"   "configs_m/aitodv2_faster_rcnn_nwdrka_softiou/nwdrka_softiou.py"

run_one "nwdrka_dyab_balanced"      "configs_m/aitodv2_faster_rcnn_nwdrka_dsl/nwdrka_dsl_balanced.py"

#
# Report
#
echo "===================="
echo "SWEEP FINISHED"
echo "===================="
echo "Total experiments : 3"
echo "Failed            : ${#FAILED[@]}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failures:"
    for f in "${FAILED[@]}"; do
        echo "  - ${f}  →  ${LOG_DIR}/${f}.log"
    done
fi
echo "Log directory     : ${LOG_DIR}"
echo "Work directories  : work_dirs/${EXP_PREFIX}/"
