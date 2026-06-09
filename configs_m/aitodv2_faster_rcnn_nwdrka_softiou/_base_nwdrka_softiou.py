"""NWD-RKA + MaxSoftIoU (simple soft-label RCNN) combined base config.

Motivation: the full DSL-DYAB RCNN head (DynamicSoftLabelAssignerDScaleDYAB
+ dynamic-k matching + cost matrix) hurt AP when stacked on top of NWD-RKA /
RFLA. This config keeps ONLY the soft-label part of the method and drops the
dynamic-assignment machinery:

  **RPN stage** (NWD-RKA, unchanged):
    * RankingAssigner with the NWD metric (topk=2).

  **RCNN stage** (MaxSoftIoU):
    * MaxSoftIoUAssigner — plain MaxIoU hard assignment (same pos/neg as the
      baseline), but the per-positive IoU is area-refine calibrated so small
      objects get a lifted soft-label ceiling.
    * SoftLabelRoIHead — StandardRoIHead that forwards that calibrated IoU to
      QualityFocalLoss as the soft classification target. It keeps the
      standard single-pass IoU assignment (NO prediction-aware two-pass
      forward), so the ONLY change vs the baseline is the soft-label loss.

All other settings inherit from ``aitodv2_faster_rcnn/_base_.py``.
"""

_base_ = ['../aitodv2_faster_rcnn_nwdrka/_base_nwdrka.py']

model = dict(
    # ── RoI Head: IoU soft target via QFL (thin StandardRoIHead subclass) ─
    roi_head=dict(
        type='SoftLabelRoIHead',
        bbox_head=dict(
            num_classes=8,
            loss_cls=dict(
                type='QualityFocalLoss',
                use_sigmoid=True,
                beta=2.0,
                loss_weight=1.0,
                custom_cls_channels=True),
        ),
    ),
    # ── RCNN Train CFG: MaxSoftIoU + standard RandomSampler ───────────────
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                _delete_=True,
                type='MaxSoftIoUAssigner',
                pos_iou_thr=0.3,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=False,
                ignore_iof_thr=-1,
                r_ref=32.0,
                calibrate_mode='add_1'),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False,
        ),
    ),
)
