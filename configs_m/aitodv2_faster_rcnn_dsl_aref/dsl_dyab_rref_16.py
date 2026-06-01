"""DSL-DScale-DYAB — ``r_ref=16`` for area-refine soft label.

Smaller r_ref ⇒ ρ = 0.5 at area ≈ 256 px² (≈ 16×16 box).
Only very tiny objects get meaningful soft-label boost.

Tests whether calibration should be concentrated on the tiniest objects.
Compare with r_ref=32 (default) and r_ref=64.
"""

_base_ = ['./dsl_dyab_default.py']

model = dict(
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(r_ref=16.0),
        ),
    ),
)
