_base_ = [
    './aitodv2_iou_r50_fpn_2x.py',
]


# Faster R-CNN with prediction-aware assignment AND IoU-aware soft cls
# target via QualityFocalLoss. The dynamic assigner picks positives based
# on (cls, iou) cost; QFL then trains cls with the IoU of those positives
# as the soft target, propagating the soft signal end-to-end.
model = dict(
    roi_head=dict(
        type='DynAssignRoIHead',
        # DynamicSoftLabelAssigner internally applies sigmoid /
        # BCEWithLogits, so it wants raw logits as `scores`.
        cls_score_activation='identity',
        # DynamicSoftLabel reads priors as (cx, cy, stride, stride);
        # convert from RPN XYXY proposals first.
        prior_format='point',
        use_iou_soft_target=True,
        bbox_head=dict(
            reg_class_agnostic=True,
            # custom_cls_channels=True: bbox_head emits num_classes sigmoid
            # channels (no dead bg channel) and inference uses sigmoid.
            loss_cls=dict(
                type='QualityFocalLoss',
                use_sigmoid=True,
                beta=2.0,
                loss_weight=1.0,
                custom_cls_channels=True),
        ),
    ),
    train_cfg=dict(
        # PseudoSampler keeps every RPN proposal as a cls training sample.
        # The base config emits 3000 proposals/image; trim to keep the
        # pos:neg ratio sane during training.
        rpn_proposal=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            _delete_=True,
            assigner=dict(
                type='DynamicSoftLabelAssigner',
                soft_center_radius=3.0,
                topk=13,
                iou_weight=3.0),
            sampler=dict(type='PseudoSampler'),
            pos_weight=-1,
            debug=False),
    ),
)
