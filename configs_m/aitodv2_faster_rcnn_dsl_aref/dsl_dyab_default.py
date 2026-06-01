"""DSL-DScale-DYAB — default hyperparameters.

Uses ``DynamicSoftLabelAssignerDScaleDYAB`` with three enhancements:

  expansion   — candidate region expanded by scale_ratio × stride (default 1.0)
  dyab        — DyabCalibrationAware: compensates soft-label calibration
  area-refine — add_1 mode, r_ref=32

DyabCalibrationAware:
  α_i = α₀ - Δα·(1-ρ_i)    cls weight decreases for small objects
  β_i = β₀ + Δβ·(1-ρ_i)    IoU weight increases for small objects
  → when soft-label is boosted for small objects, the assigner relies
    more on IoU (less on compressed cls scores) to discriminate.
"""

_base_ = ['./dsl_aref_default.py']

model = dict(
    train_cfg=dict(
        rcnn=dict(
            _delete_=True,
            assigner=dict(
                type='DynamicSoftLabelAssignerDScaleDYAB',
                soft_center_radius=3.0,
                topk=13,
                iou_weight=3.0,
                # ── area-refine soft label ──
                r_ref=32.0,
                calibrate_mode='add_1',
                # ── expansion (scale_ratio × stride) ──
                scale_ratio=1.0,
                # ── dyab ──
                dyab_type='DyabCalibrationAware',
                dyab_kwargs=dict(
                    alpha_base=1.0,
                    beta_base=4.0,
                    delta_alpha=0.5,
                    delta_beta=2.0,
                    r_ref=64.0,
                ),
            ),
            sampler=dict(type='PseudoSampler'),
            pos_weight=-1,
            debug=False,
        ),
    ),
)
