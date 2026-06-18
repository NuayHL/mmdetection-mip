"""DSL-DYAB-RCNN — size-dependent expansion on Detectors.

expansion_type='size_dependent': margin = scale_ratio × (1-ρ) × stride

Applied identically across all 3 cascade stages.
"""

_base_ = ['./dsl_dyab_rcnn_default.py']

model = dict(
    train_cfg=dict(
        rcnn=[
            dict(assigner=dict(expansion_type='size_dependent')),
            dict(assigner=dict(expansion_type='size_dependent')),
            dict(assigner=dict(expansion_type='size_dependent')),
        ],
    ),
)
