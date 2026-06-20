#!/bin/bash

# ============================================================
# AITOD-v2  Faster R-CNN  USAA (TaskAlignedAssignerUSAA) — sweep
#
# RCNN = faithful TAL port (multiplicative s^α×u^β, TAL soft label,
# dmetric/dscale/area-refine/dyab) on top of three RPNs:
#   * baseline MaxIoU
#   * RFLA (KLD / WD)
#   * NWD-RKA
#
# Run from repo root:
#   bash configs_m/aitodv2_faster_rcnn_usaa_tal/run_all_usaa.sh
# ============================================================
set -euo pipefail

EXP_PREFIX="aitodv2_faster_rcnn_usaa_tal"
CFG_DIR="configs_m/${EXP_PREFIX}"

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
# (2) baseline RPN + USAA
#
run_one "usaa_tal"       "${CFG_DIR}/usaa_tal.py"

#
# (4) RFLA RPN + USAA  (KLD / WD)
#
run_one "rfla_usaa_kld"  "${CFG_DIR}/rfla_usaa_kld.py"
run_one "rfla_usaa_wd"   "${CFG_DIR}/rfla_usaa_wd.py"

#
# (6) NWD-RKA RPN + USAA
#
run_one "nwdrka_usaa"    "${CFG_DIR}/nwdrka_usaa.py"

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