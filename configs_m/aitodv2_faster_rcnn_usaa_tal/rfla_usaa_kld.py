"""RFLA-KLD RPN + USAA Task-Aligned RCNN  (the 1 + 1 ≥ 2 target).

RPN:   RFLA (RFGenerator + HieAssigner with KL-divergence metric, topk=[3,1],
       ratio=0.9) — widens / re-shapes the tiny-object proposal pool,
       IoU-independent.
RCNN:  TaskAlignedAssignerUSAA (inherited from ``_base_usaa.py``) — dynamically
       decides which proposals are good positives from prediction quality
       (s^α × u^β), with a properly normalized, area-refined TAL soft label.

Currency option: to align the RCNN soft-target metric with RFLA's KLD RPN,
set ``align_mode='kl'`` / ``score_mode='kl'`` on the assigner (see
``_base_usaa.py``). Default is ``iou`` (prediction-quality driven).
"""

_base_ = ['./_base_usaa.py']

# RFLA RPN override only — USAA RCNN inherited unchanged from _base_usaa.py.
model = dict(
    rpn_head=dict(
        anchor_generator=dict(
            type='RFGenerator',
            fpn_layer='p2',
            fraction=0.5,
            strides=[4, 8, 16, 32, 64]),
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                _delete_=True,
                type='HieAssigner',
                ignore_iof_thr=-1,
                gpu_assign_thr=512,
                iou_calculator=dict(type='BboxDistanceMetric'),
                assign_metric='kl',
                topk=[3, 1],
                ratio=0.9),
        ),
    ),
)