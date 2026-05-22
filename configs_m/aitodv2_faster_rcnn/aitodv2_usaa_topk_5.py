"""USAA — ``DynamicSoftLabelAssigner.topk = 5`` (default 13).

Lower top-k → fewer candidate priors per GT, tighter selection.
"""

_base_ = ['./aitodv2_usaa.py']

model = dict(
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(topk=5),
        ),
    ),
)
