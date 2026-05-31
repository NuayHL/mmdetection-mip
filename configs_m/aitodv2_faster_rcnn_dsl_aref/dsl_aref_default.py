"""DSL-AreaRefine — default hyperparameters.

Same DynAssignRoIHead + QualityFocalLoss setup as USAA, but with
``DynamicSoftLabelAssignerAreaRefine`` instead of vanilla
``DynamicSoftLabelAssigner``.

Area-refine soft label calibration:
  rho   = area / (area + r_ref²)
  small objects (area ≪ r_ref²):  rho → 0  → soft-label IoU boosted toward 1
  large objects (area ≫ r_ref²):  rho → 1  → soft-label IoU unchanged

Default settings:
  calibrate_mode = 'add_1'   (iou + (1-rho)·iou·(1-iou))
  r_ref          = 32        (medium transition point)
"""

_base_ = ['./_base_.py']

model = dict(
    roi_head=dict(
        type='DynAssignRoIHead',
        cls_score_activation='identity',
        prior_format='point',
        use_iou_soft_target=True,
        bbox_head=dict(
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
            assigner=dict(
                type='DynamicSoftLabelAssignerAreaRefine',
                soft_center_radius=3.0,
                topk=13,
                iou_weight=3.0,
                r_ref=32.0,
                calibrate_mode='add_1'),
            sampler=dict(type='PseudoSampler'),
            pos_weight=-1,
            debug=False),
    ),
)
