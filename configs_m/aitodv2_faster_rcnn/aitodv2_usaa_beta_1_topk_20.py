"""USAA — ``DynamicSoftLabelAssigner.topk = 20`` (default 13).

Higher top-k → more candidate priors per GT, looser selection.
"""

_base_ = ['./aitodv2_usaa.py']

model = dict(
    roi_head=dict(
        bbox_head=dict(
            loss_cls=dict(beta=1.0),
        ),
    ),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(topk=20),
        ),
    ),
)
