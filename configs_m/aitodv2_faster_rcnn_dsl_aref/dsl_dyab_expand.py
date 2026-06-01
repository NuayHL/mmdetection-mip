"""DSL-DScale-DYAB — ``scale_ratio=2.0`` (aggressive expansion).

Doubles the expansion margin to 2× stride.  For stride=8, centres can
be up to 16 px outside the GT box; for stride=64, up to 128 px.

Tests whether more aggressive expansion further helps small objects.
Compare with scale_ratio=1.0 (default) and scale_ratio=0.0.
"""

_base_ = ['./dsl_dyab_default.py']

model = dict(
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(scale_ratio=2.0),
        ),
    ),
)
