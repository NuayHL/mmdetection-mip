"""DSL-AreaRefine — ``r_ref=16`` (vs default 32).

Smaller r_ref ⇒ rho = 0.5 at area ≈ 256 px² (≈ 16×16 box).
More objects cross the transition threshold → calibration applied more broadly.

Tests whether boosting should be concentrated on the tiniest objects (r_ref=16)
or spread across more objects (r_ref=32, default).
"""

_base_ = ['./dsl_aref_default.py']

model = dict(
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(r_ref=16.0),
        ),
    ),
)
