"""RFLA + DSL-DYAB combined base config.

Applies both enhancements to the AITOD-v2 Faster R-CNN baseline:

  **RPN stage** (from RFLA):
    * RFGenerator anchors — effective receptive field as anchor prior
    * HieAssigner with KLD — hierarchical label assignment for proposals

  **RCNN stage** (from DSL-DYAB):
    * DynAssignRoIHead — dynamic label assignment head
    * QualityFocalLoss — IoU-aware classification loss
    * DynamicSoftLabelAssignerDScaleDYAB with DyabDSL — dynamic α/β
      cost weighting + expansion + area-refine soft label calibration

All other settings (dataset, schedule, backbone, neck, test config) are
identical to the ``aitodv2_faster_rcnn/_base_.py`` baseline.
"""

_base_ = ['../aitodv2_faster_rcnn_rfla/_base_rfla.py']

model = dict(
    # ── RoI Head: DSL ────────────────────────────────────────────────────
    roi_head=dict(
        type='DynAssignRoIHead',
        cls_score_activation='identity',
        prior_format='point',
        use_iou_soft_target=True,
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
    # ── RCNN Train CFG: DSL-DYAB ─────────────────────────────────────────
    train_cfg=dict(
        rcnn=dict(
            _delete_=True,
            assigner=dict(
                type='DynamicSoftLabelAssignerDScaleDYAB',
                soft_center_radius=2.0,
                topk=13,
                iou_weight=3.0,
                r_ref=32.0,
                calibrate_mode='add_1',
                scale_ratio=1.0,
                expansion_type='static',
                expansion_r_ref=32.0,
                dyab_type='DyabDSL',
                dyab_kwargs=dict(
                    alpha_base=0.8,
                    beta_base=1.5,
                    delta_alpha=0.4,
                    delta_beta=0.5,
                    r_ref=64.0,
                ),
            ),
            sampler=dict(type='PseudoSampler'),
            pos_weight=-1,
            debug=False,
        ),
    ),
)
