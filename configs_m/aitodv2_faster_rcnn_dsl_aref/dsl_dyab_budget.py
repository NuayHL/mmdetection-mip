"""DSL-DScale-DYAB — ``dyab_type='DyabBudgetShift'`` (vs DyabCalibrationAware).

DyabBudgetShift: α + β = const  (budget-preserving)

  α_i = α₀ + Δ·(1-ρ_i)    cls weight increases for small objects
  β_i = β₀ - Δ·(1-ρ_i)    IoU weight decreases for small objects

Intuition: for small objects, IoU is noisy → rely more on cls score
for matching.  Opposite philosophy from DyabCalibrationAware.
"""

_base_ = ['./dsl_dyab_default.py']

model = dict(
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                _delete_=True,
                type='DynamicSoftLabelAssignerDScaleDYAB',
                soft_center_radius=3.0,
                topk=13,
                iou_weight=3.0,
                r_ref=32.0,
                calibrate_mode='add_1',
                scale_ratio=1.0,
                dyab_type='DyabBudgetShift',
                dyab_kwargs=dict(
                    alpha_base=1.0,
                    beta_base=4.0,
                    delta=1.5,
                    r_ref=32.0,
                ),
            ),
            sampler=dict(type='PseudoSampler'),
            pos_weight=-1,
            debug=False,
        ),
    ),
)
