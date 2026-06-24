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

_base_ = ['./rfla_predsoftlabel.py']

model = dict(
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                score_mode='iou',
                calibrate_mode='none',
                floor_to_pos_thr=False),
        ),
    ),
)
