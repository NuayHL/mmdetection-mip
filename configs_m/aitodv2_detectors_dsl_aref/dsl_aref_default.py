"""DSL-AreaRefine — default hyperparameters (Detectors, 3-stage DSL).

Applies :class:`DynAssignCascadeRoIHead` on the DetectoRS+RFP backbone —
keeps the cascade refinement mechanism, but replaces the standard
MaxIoUAssigner + CrossEntropyLoss with DynamicSoftLabelAssignerAreaRefine
+ QualityFocalLoss in every stage.

Area-refine soft label calibration:
  rho   = area / (area + r_ref²)

Default settings:
  calibrate_mode = 'add_1'
  r_ref          = 32
"""

_base_ = ['./_base_.py']

model = dict(
    roi_head=dict(
        type='DynAssignCascadeRoIHead',
        cls_score_activation='identity',
        prior_format='point',
        use_iou_soft_target=True,
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='QualityFocalLoss',
                    use_sigmoid=True,
                    beta=2.0,
                    loss_weight=1.0,
                    custom_cls_channels=True),
            ),
            dict(
                type='Shared2FCBBoxHead',
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='QualityFocalLoss',
                    use_sigmoid=True,
                    beta=2.0,
                    loss_weight=1.0,
                    custom_cls_channels=True),
            ),
            dict(
                type='Shared2FCBBoxHead',
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='QualityFocalLoss',
                    use_sigmoid=True,
                    beta=2.0,
                    loss_weight=1.0,
                    custom_cls_channels=True),
            ),
        ],
    ),
    train_cfg=dict(
        rcnn=[
            dict(
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
            dict(
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
            dict(
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
        ],
    ),
)
