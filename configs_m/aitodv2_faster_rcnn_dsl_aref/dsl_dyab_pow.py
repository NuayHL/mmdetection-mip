"""DSL-DScale-DYAB — ``calibrate_mode='pow'`` (vs default 'add_1').

Pow mode:  soft_iou = iou^rho
  - More aggressive boost for very small objects (rho → 0 ⇒ iou⁰ → 1)
  - Smoother gradient than add_1 at intermediate rho

When combined with DyabCalibrationAware, the stronger calibration in pow
mode should be compensated by larger δβ in the dyab module.

This variant isolates the calibration function shape.
"""

_base_ = ['./dsl_dyab_default.py']

model = dict(
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(calibrate_mode='pow'),
        ),
    ),
)
