"""USAA — ``DynamicSoftLabelAssigner.iou_weight = 1.0`` (default 3.0).

Lower iou_weight → cost matrix relies more on the classification term.
"""

_base_ = ['./aitodv2_usaa.py']

model = dict(
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(iou_weight=1.0),
        ),
    ),
)
