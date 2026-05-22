"""USAA — ``DynamicSoftLabelAssigner.soft_center_radius = 1.0`` (default 3.0).

Smaller radius → stronger penalty for priors whose center is far from
the GT center.
"""

_base_ = ['./aitodv2_usaa.py']

model = dict(
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(soft_center_radius=1.0),
        ),
    ),
)
