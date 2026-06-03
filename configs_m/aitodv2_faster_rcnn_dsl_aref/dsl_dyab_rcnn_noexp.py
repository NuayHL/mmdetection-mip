"""DSL-DYAB-RCNN — no expansion (scale_ratio=0).

Disables candidate expansion entirely while keeping DyabDSL.
Isolates dyab's contribution without expansion confounding.

Compare with default to measure expansion's impact.
"""

_base_ = ['./dsl_dyab_rcnn_default.py']

model = dict(
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(scale_ratio=0.0),
        ),
    ),
)
