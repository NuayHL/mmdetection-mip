"""NWD-RKA RPN + USAA Task-Aligned RCNN.

RPN:   NWD-RKA (RankingAssigner with Normalized Wasserstein Distance, top-2
       per GT; standard AnchorGenerator — NWD-RKA does NOT change anchors).
RCNN:  TaskAlignedAssignerUSAA (inherited from ``_base_usaa.py``).

Currency option: to align the RCNN soft-target metric with the NWD RPN, set
the assigner ``align_mode='nwd'`` / ``score_mode='nwd'`` (see ``_base_usaa.py``).
Default is ``iou`` (prediction-quality driven).
"""

_base_ = ['./_base_usaa.py']

# NWD-RKA RPN override only — USAA RCNN inherited unchanged from _base_usaa.py.
model = dict(
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                _delete_=True,
                type='RankingAssigner',
                ignore_iof_thr=-1,
                gpu_assign_thr=512,
                iou_calculator=dict(type='BboxDistanceMetric', constant=12.7),
                assign_metric='nwd',
                topk=2),
        ),
    ),
)