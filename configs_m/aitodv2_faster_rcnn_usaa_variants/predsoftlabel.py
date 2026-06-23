"""Plain Faster R-CNN + prediction-aware soft-label RCNN (USAA variant D).

The minimal *task-aligned* softening of the baseline MaxIoU RCNN. Compared with
``softlabel_roi.py`` (MaxSoftIoUAssigner: calibrated **proposal** IoU, single
pass) the ONLY conceptual change is the soft-label source:

  * MaxIoUPredSoftLabelAssigner — the hard pos/neg selection is byte-for-byte
    the baseline MaxIoU (thresholds 0.5, match_low_quality=False), so the part
    that already combines with an RFLA / NWD-RKA RPN is left untouched. Only the
    positive cls target becomes
        soft = max( calibrate(IoU(decoded_pred_bbox, matched_gt)), pos_iou_thr )
    i.e. it reflects the head's **actual** localisation quality (task-aligned),
    floored at pos_iou_thr so a poor early prediction can never collapse the
    positive gradient (the "no big debuff" guarantee).
  * DynAssignRoIHead — needed for the two-pass forward that exposes the decoded
    prediction to the assigner. cls_score_activation is irrelevant to this
    assigner (it ignores scores) but sigmoid is required by QFL.
  * QualityFocalLoss consumes the soft target via use_iou_soft_target=True.

Compare against the baseline aitodv2_iou.py (mAP 0.132) and against
softlabel_roi.py to isolate prediction-IoU(+floor) vs proposal-IoU.
"""

_base_ = ['../aitodv2_faster_rcnn/_base_.py']

model = dict(
    roi_head=dict(
        type='DynAssignRoIHead',
        cls_score_activation='sigmoid',   # required by QFL (scores unused here)
        prior_format='xyxy',              # super().assign reads priors as boxes
        use_iou_soft_target=True,         # deliver soft label to QFL
        cls_avg_factor='num_samples',     # stable with bounded RandomSampler
        bbox_head=dict(
            num_classes=8,
            reg_class_agnostic=False,
            loss_cls=dict(
                type='QualityFocalLoss',
                use_sigmoid=True,
                beta=0.0,
                loss_weight=1.0,
                custom_cls_channels=True),
        ),
    ),
    train_cfg=dict(
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
                floor_to_pos_thr=True),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.3,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
        ),
    ),
)
