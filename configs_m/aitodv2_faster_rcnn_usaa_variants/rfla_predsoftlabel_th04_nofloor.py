"""RFLA-KLD RPN + prediction-aware soft-label RCNN  (the 1 + 1 ≥ 2 target).

RPN:   RFLA (RFGenerator + HieAssigner, KL metric, topk=[3,1], ratio=0.9) —
       widens / re-shapes the tiny-object proposal pool, IoU-independent.
RCNN:  MaxIoUPredSoftLabelAssigner (inherited from ``predsoftlabel.py``) — keeps
       the baseline MaxIoU hard assignment (proven to combine with RFLA) and
       only softens the positive cls target to the *floored, calibrated*
       prediction IoU.

Currency note: the RCNN soft label is computed from the **decoded prediction**,
not the proposal, so unlike the additive DSL port it does not collapse on the
near-zero-CIoU tiny proposals that RFLA surfaces — the floor (pos_iou_thr) plus
the calibration keep the target meaningful. ``score_mode='iou'`` is the default;
switch to ``'nwd'`` / ``'kl'`` to align the RCNN soft-target currency with the
KLD RPN (separate ablation).
"""

_base_ = ['./predsoftlabel.py']

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
        rcnn=dict(
            assigner=dict(
                _delete_=True,
                type='MaxIoUPredSoftLabelAssigner',
                # ── hard assignment: IDENTICAL to the baseline RCNN MaxIoU ──
                pos_iou_thr=0.4,
                neg_iou_thr=0.4,
                min_pos_iou=0.4,
                match_low_quality=False,
                ignore_iof_thr=-1,
                # ── soft label: calibrated prediction IoU, floored ──
                r_ref=32.0,
                calibrate_mode='add_1',
                lambda_refine=1.0,
                score_mode='iou',
                floor_to_pos_thr=False),
        ),
    ),
)
