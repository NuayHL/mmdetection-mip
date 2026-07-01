"""DINO-4scale R50 + USAA flavour on AITOD-v2 (12e).

A faithful mmdetection port of the RT-DETR USAA variant
(``Ultralytics/cfg/rt-detr/rtdetr-resnet50-usaa.yaml`` /
``RTDETRDetectionLoss_USAA``).  Two changes vs the baseline
``dino-4scale_r50_1xb2-12e_aitodv2.py``:

  Change 1 — ASSIGNMENT (scale-aware cost reweighting)
      HungarianAssigner → HungarianAssignerUSAA.  Per-GT, the classification
      matching cost is reduced and the spatial (L1 + GIoU) matching cost is
      boosted for small objects, driven by ρ = area / (area + r_ref_ab²).

  Change 2 — SUPERVISION (calibrated IoU soft label)
      FocalLoss (hard label)  → QualityFocalLoss (IoU soft label), and that IoU
      target is area-refine calibrated by DINOHeadUSAA:
          f(u,ρ) = u + (1 − ρ)·u·(1 − u),   ρ = area / (area + r_ref_cal²)
      so small objects are not systematically under-supervised.  The DINO code
      already computes IoU(pred, gt) as the QFL target for positives; USAA only
      calibrates it.  The calibration is applied to the matching-query,
      encoder, and denoising losses alike.

Compare against ``dino-4scale_r50_1xb2-12e_aitodv2.py`` (the baseline).

Ablation knobs (to isolate each USAA component):
  * matcher only  : set bbox_head.loss_cls back to the baseline FocalLoss
                    (dict(type='FocalLoss', use_sigmoid=True, gamma=2.0,
                    alpha=0.25, loss_weight=1.0)) — Change 1 alone.
  * supervision   : set train_cfg.assigner.scale_aware=False — Change 2 alone
                    (raw-IoU vs calibrated: bbox_head.use_soft_label_cal=False).
  * cost strength : r_ref_ab / cls_reduction / spatial_boost.
  * label lift    : r_ref_cal / cal_type ('add_1' | 'pow').
"""

_base_ = ['./dino-4scale_r50_1xb2-12e_aitodv2.py']

model = dict(
    bbox_head=dict(
        type='DINOHeadUSAA',
        # ── USAA Change 2: calibrated IoU soft label ──
        use_soft_label_cal=True,
        r_ref_cal=32.0,
        cal_type='add_1',
        calibrate_dn=True,
        # QFL exposes the IoU soft-label supervision path (mmdet's VFL analogue).
        loss_cls=dict(
            _delete_=True,
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0)),
    train_cfg=dict(
        # ── USAA Change 1: scale-aware one-to-one matching ──
        assigner=dict(
            _delete_=True,
            type='HungarianAssignerUSAA',
            r_ref_ab=64.0,
            cls_reduction=0.5,
            spatial_boost=0.5,
            scale_aware=True,
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])))
