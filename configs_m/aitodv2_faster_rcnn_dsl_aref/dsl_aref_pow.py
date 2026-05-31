"""DSL-AreaRefine — ``calibrate_mode='pow'`` (vs default 'add_1').

Pow mode:  soft_iou = iou^rho
- more aggressive boost for very small objects (rho → 0 ⇒ iou⁰ → 1)
- smoother gradient than add_1 at intermediate rho

This isolates the effect of the calibration function shape.
"""

_base_ = ['./dsl_aref_default.py']

model = dict(
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(calibrate_mode='pow'),
        ),
    ),
)
