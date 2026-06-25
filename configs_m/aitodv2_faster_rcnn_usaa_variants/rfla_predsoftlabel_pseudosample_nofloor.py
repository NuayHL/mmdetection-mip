"""RFLA-KLD RPN + prediction-aware soft-label RCNN, honest IoU  (Branch A).

Strategy A — make the soft label a *faithful* estimate of localisation quality
(no inflation), optimising mAP + ECE + high-IoU AP. This trades back some of
the AP50 gain of ``rfla_predsoftlabel.py`` in exchange for honest ranking.

The cls target for a positive becomes simply ``IoU(decoded_pred, matched_gt)``
(clamped to [0,1]):
  * score_mode='iou'      — honest localisation quality.
  * calibrate_mode='none' — no add_1 inflation → removes an over-confidence
    source (better ECE, sharper high-IoU ranking).
  * floor_to_pos_thr=False — let poorly-localised positives keep their honest
    low target; this is the main lever that was capping mAP. Safe because
    add_gt_as_proposals=True still gives every GT a target=1.0 positive.

With QFL(beta=0) (pure soft-BCE) the predicted score converges to the IoU →
well-calibrated by construction. Compare against ``rfla_predsoftlabel_nwd.py``
(Branch B) to decide whether the tiny-object regime prefers an NWD or an IoU
currency.
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
            sampler=dict(type='PseudoSampler'),
        ),
        rcnn=dict(
            assigner=dict(
                _delete_=True,
                type='MaxIoUPredSoftLabelAssigner',
                # ── hard assignment: IDENTICAL to the baseline RCNN MaxIoU ──
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1,
                # ── soft label: calibrated prediction IoU, floored ──
                r_ref=32.0,
                calibrate_mode='add_1',
                lambda_refine=1.0,
                score_mode='iou',
                floor_to_pos_thr=False),
            sampler=dict(type='PseudoSampler'),
        ),
    ),
)