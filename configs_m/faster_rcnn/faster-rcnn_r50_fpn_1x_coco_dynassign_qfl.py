_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
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
        use_iou_soft_target=True,
        bbox_head=dict(
            reg_class_agnostic=True,
            # Sigmoid head: cls output has num_classes channels (no bg).
            loss_cls=dict(
                type='QualityFocalLoss',
                use_sigmoid=True,
                beta=2.0,
                loss_weight=1.0),
        ),
    ),
    train_cfg=dict(
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
