#!/bin/bash
# ============================================================
# AITOD-v2  DSL-DYAB-RCNN size-dependent expansion
# Cascade R-CNN & Detectors — baseline vs DSL-adapt comparison
#
# Experiments:
#   1. Cascade R-CNN  baseline     (CascadeRoIHead + MaxIoUAssigner)
#   2. Cascade R-CNN  + DSL-adapt  (DynAssignCascadeRoIHead + DScaleDYAB)
#   3. Detectors       baseline     (CascadeRoIHead + MaxIoUAssigner)
#   4. Detectors       + DSL-adapt  (DynAssignCascadeRoIHead + DScaleDYAB)
#
# Both DSL variants use the 3-stage cascade refinement internally —
# the only change is assigner (DSL vs MaxIoU) + sampler (PseudoSampler vs
# RandomSampler) + loss (QFL vs CrossEntropy).
#
# Run from repo root:
#   bash configs_m/run_all_dsl_dyab_adapt.sh
# ============================================================
set -euo pipefail

LOG_DIR="terminal_log/terminal_log_dsl_dyab_adapt"
mkdir -p "$LOG_DIR"

FAILED=()

run_one() {
    local exp_name="$1"
    local config_path="$2"
    local work_dir="$3"

    echo "=============================================="
    echo "  Experiment : ${exp_name}"
    echo "  Config     : ${config_path}"
    echo "  Log        : ${LOG_DIR}/${exp_name}.log"
    echo "  Started at : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================="

    if python tools/train.py "${config_path}" \
        --work-dir "${work_dir}" \
        > "${LOG_DIR}/${exp_name}.log" 2>&1; then
        echo "  [PASS]  ${exp_name}  finished  $(date '+%Y-%m-%d %H:%M:%S')"
    else
        echo "  [FAIL]  ${exp_name}  failed    $(date '+%Y-%m-%d %H:%M:%S')"
        FAILED+=("${exp_name}")
    fi
    echo ""
}

#
# Experiment 1 — Cascade R-CNN baseline (CascadeRoIHead + MaxIoUAssigner)
#
run_one \
    "cascade_baseline" \
    "configs_m/cascade_rcnn/aitodv2_iou_cascade_rcnn_2x.py" \
    "./work_dirs/aitodv2_cascade_rcnn/baseline"

#
# Experiment 2 — Cascade R-CNN + DSL-DYAB size-dep
#   DynAssignCascadeRoIHead — same 3-stage refinement, DSL assigner
#
run_one \
    "cascade_dsl_dyab_adapt" \
    "configs_m/aitodv2_cascade_rcnn_dsl_aref/dsl_dyab_rcnn_adapt.py" \
    "./work_dirs/aitodv2_cascade_rcnn_dsl_aref/dsl_dyab_rcnn_adapt"

#
# Experiment 3 — Detectors baseline (CascadeRoIHead + MaxIoUAssigner)
#
run_one \
    "detectors_baseline" \
    "configs_m/detectors/aitodv2_iou_detectors_cascade-rcnn_r50_2x.py" \
    "./work_dirs/aitodv2_detectors/baseline"

#
# Experiment 4 — Detectors + DSL-DYAB size-dep
#
run_one \
    "detectors_dsl_dyab_adapt" \
    "configs_m/aitodv2_detectors_dsl_aref/dsl_dyab_rcnn_adapt.py" \
    "./work_dirs/aitodv2_detectors_dsl_aref/dsl_dyab_rcnn_adapt"

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
echo ""
echo "Comparison matrix (same cascade refinement, different assigners):"
echo ""
echo "  Experiment                RoI Head                     Assigner"
echo "  ------------------------  ---------------------------  ------------------------"
echo "  cascade_baseline          CascadeRoIHead               MaxIoUAssigner"
echo "  cascade_dsl_dyab_adapt    DynAssignCascadeRoIHead      DScaleDYAB size-dep"
echo "  detectors_baseline        CascadeRoIHead               MaxIoUAssigner"
echo "  detectors_dsl_dyab_adapt  DynAssignCascadeRoIHead      DScaleDYAB size-dep"
echo ""
echo "Within each backbone pair, the only differences are:"
echo "  assigner, sampler, loss function"
echo "The cascade refinement mechanism is preserved in all 4 experiments."
