"""DSL-DScale-DYAB — ``scale_ratio=0.0`` (no expansion, strict containment).

Turns off candidate expansion entirely.  Equivalent to the original
DynamicSoftLabelAssignerAreaRefine but with DyabCalibrationAware active.

Tests whether expansion (scale_ratio=1.0) helps vs strict containment.
"""

_base_ = ['./dsl_dyab_default.py']

model = dict(
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(scale_ratio=0.0),
        ),
    ),
)
