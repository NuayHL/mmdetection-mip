"""DSL-AreaRefine — ``r_ref=64`` (vs default 32).

Larger r_ref ⇒ rho = 0.5 at area ≈ 4096 px² (≈ 64×64 box).
Only very small objects (relative to this reference) get meaningful boost.

Tests whether the calibration should be applied to a wider size range
or reserved only for truly tiny objects.
"""

_base_ = ['./dsl_aref_default.py']

model = dict(
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(r_ref=64.0),
        ),
    ),
)
