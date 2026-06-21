"""USAA Task-Aligned RCNN base (standalone on the AITOD-v2 Faster R-CNN baseline).

RPN:   standard AnchorGenerator + MaxIoUAssigner (inherited baseline, unchanged).
RCNN:  TaskAlignedAssignerUSAA (faithful TAL port) + DynAssignRoIHead + QFL.

This file IS the standalone experiment — it should improve over the MaxIoU
baseline on its own. The RFLA-combined variant (``rfla_usaa_tal.py``) inherits
this file and adds ONLY the RPN override, so the USAA RCNN block below is
written exactly once.

RCNN stage:
  * DynAssignRoIHead — two-pass prediction-aware head. ``cls_score_activation
    ='sigmoid'`` so the assigner sees probabilities ``s ∈ [0,1]`` (required by
    the multiplicative ``s^α × u^β`` metric). ``use_iou_soft_target=True``
    forwards ``AssignResult.max_overlaps`` (the TAL-normalized soft label) to
    QualityFocalLoss.
  * TaskAlignedAssignerUSAA — multiplicative metric, fixed top-k, TAL
    soft-label normalization, dmetric (CIoU), dscale, area-refine, dyab.

All assigner hyperparameters mirror the USAA standard YAML
``Ultralytics/cfg/usaa/yolo12_usaa_raw_dyabcalra64_ra32_rtadd_s10.yaml``.
"""

_base_ = ['../aitodv2_faster_rcnn/_base_.py']

model = dict(
    roi_head=dict(
        type='DynAssignRoIHead',
        cls_score_activation='sigmoid',   # s must be a probability for s^α
        prior_format='point',             # (cx, cy, min(w,h), min(w,h))
        use_iou_soft_target=True,         # deliver TAL soft label to QFL
        # With RandomSampler (bounded 512 samples, 1:3) 'num_samples' is both
        # stable and undiluted — avg_factor≈512 with a healthy positive
        # fraction, NOT the ~2000 of PseudoSampler. Do NOT use 'pos_soft_sum'
        # here: its denominator Σ(soft targets) collapses to ~0.01 on sparse
        # batches while ~2000 bg proposals stay in the numerator → loss spikes
        # to 1000+ → logits run away → NaN (this is exactly the failure you saw).
        cls_avg_factor='num_samples',
        bbox_head=dict(
            num_classes=8,
            reg_class_agnostic=True,
            loss_cls=dict(
                type='QualityFocalLoss',
                use_sigmoid=True,
                beta=2.0,
                loss_weight=1.0,
                custom_cls_channels=True),
        ),
    ),
    train_cfg=dict(
        rcnn=dict(
            _delete_=True,
            # ── USAA standard ────────────────────────────────────────────
            # Mirrors Ultralytics cfg/usaa/
            #   yolo12_usaa_raw_dyabcalra64_ra32_rtadd_s10.yaml
            # (TaskAlignedAssigner_dyab_dmetric_dscale_RefineArea).
            assigner=dict(
                type='TaskAlignedAssignerUSAA',
                topk=10,
                iou_calculator=dict(type='BboxDistanceMetric'),
                # dmetric (SimD): CIoU for all three roles (USAA standard).
                # Switch align/score to 'nwd'/'kl' to match an NWD/KLD RPN's
                # currency (separate ablation, not the standard).
                overlap_mode='ciou',
                align_mode='ciou',
                score_mode='ciou',
                # dscale: static candidate expansion, scale_ratio=1.0.
                scale_ratio=1.0,
                expansion_type='static',
                expansion_r_ref=32.0,
                # area-refine soft-label ceiling: r_ref=32, add_1 (rtadd).
                r_ref=32.0,
                calibrate_mode='add_1',
                lambda_refine=1.0,
                # dyab = DyabCalibrationAware (dyabcalra64). Small objects get
                #   α↓, β↑  (less cls weight, sharper IoU ranking) to offset
                #   the area-refine ceiling boost:
                #     large obj (ρ→1): α=1.0, β=4.0
                #     tiny  obj (ρ→0): α=0.5, β=6.0
                dyab_type='DyabCalibrationAware',
                dyab_kwargs=dict(
                    alpha_base=1.0,
                    beta_base=4.0,
                    delta_alpha=0.5,
                    delta_beta=2.0,
                    r_ref=64.0,
                ),
            ),
            # RandomSampler, NOT PseudoSampler. The YOLO "all anchors" recipe
            # does not transfer to two-stage RCNN: PseudoSampler floods the QFL
            # with ~2000 background proposals (huge numerator) while the soft
            # denominator (Σ soft targets / #pos) collapses on sparse-GT
            # batches → loss spikes → logits run away → NaN ('pos_soft_sum'),
            # or avg_factor≈2000 dilutes the positive signal → AP collapse
            # ('num_samples'). RandomSampler caps negatives (1:3) and fixes the
            # denominator, curing BOTH failure modes. add_gt_as_proposals
            # guarantees every GT a positive (key for tiny objects).
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