"""USAA — ``DynamicSoftLabelAssigner.iou_weight = 5.0`` (default 3.0).

Higher iou_weight → assignment dominated by predicted IoU quality.
"""

_base_ = ['./aitodv2_usaa.py']

model = dict(
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(iou_weight=5.0),
        ),
    ),
)
