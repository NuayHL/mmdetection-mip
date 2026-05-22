"""USAA — ``DynamicSoftLabelAssigner.soft_center_radius = 5.0`` (default 3.0).

Larger radius → softer center prior, more distant priors stay competitive.
"""

_base_ = ['./aitodv2_usaa.py']

model = dict(
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(soft_center_radius=5.0),
        ),
    ),
)
