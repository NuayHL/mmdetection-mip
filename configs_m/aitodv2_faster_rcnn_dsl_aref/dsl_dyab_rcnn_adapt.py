"""DSL-DYAB-RCNN — size-dependent expansion.

expansion_type='size_dependent': margin = scale_ratio × (1-ρ) × stride
  - Small objects (ρ→0): full expansion
  - Large objects (ρ→1): near strict containment

Tests whether size-dependent expansion recovers AP medium
(vs static expansion which can over-expand medium objects).
"""

_base_ = ['./dsl_dyab_rcnn_default.py']

model = dict(
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                expansion_type='size_dependent',
            ),
        ),
    ),
)
