"""DSL-DYAB-RCNN — size-dependent expansion on Cascade R-CNN.

expansion_type='size_dependent': margin = scale_ratio × (1-ρ) × stride
  - Small objects (ρ→0): full expansion
  - Large objects (ρ→1): near strict containment

Applied identically across all 3 cascade stages.
"""

_base_ = ['./dsl_dyab_rcnn_default.py']

model = dict(
    train_cfg=dict(
        rcnn=[
            dict(assigner=dict(
                type='DynamicSoftLabelAssignerDScaleDYAB',
                expansion_type='size_dependent')),
            dict(assigner=dict(
                type='DynamicSoftLabelAssignerDScaleDYAB',
                expansion_type='size_dependent')),
            dict(assigner=dict(
                type='DynamicSoftLabelAssignerDScaleDYAB',
                expansion_type='size_dependent')),
        ],
    ),
)
