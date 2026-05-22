"""USAA variant — SimOTAAssigner instead of DynamicSoftLabelAssigner.

Both expect point-format priors (cx, cy, stride, stride), so
``prior_format='point'`` is inherited unchanged from
``aitodv2_usaa.py``.

SimOTA reads ``pred_instances.scores`` as probabilities (its cost uses
``F.binary_cross_entropy`` without logits), so switch the activation
from 'identity' (which DynamicSoftLabel wants) to 'sigmoid'.
"""

_base_ = ['./aitodv2_usaa.py']

model = dict(
    roi_head=dict(
        cls_score_activation='sigmoid',
    ),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                _delete_=True,
                type='SimOTAAssigner',
                center_radius=2.5,
                candidate_topk=10,
                iou_weight=3.0,
                cls_weight=1.0),
        ),
    ),
)
