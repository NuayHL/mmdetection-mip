_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

# Faster R-CNN with prediction-aware label assignment in the R-CNN stage.
# RPN is unchanged (MaxIoUAssigner). Only the rcnn stage uses SimOTA.
model = dict(
    roi_head=dict(
        type='DynAssignRoIHead',
        # Softmax→drop-bg is the right activation for the default CE head.
        cls_score_activation='softmax',
        # SimOTA reads priors as (cx, cy, stride_x, stride_y); convert
        # from RPN XYXY proposals before the assigner sees them.
        prior_format='point',
        use_iou_soft_target=False,
        # class-agnostic regression so we can decode (N, 4) for the assigner.
        bbox_head=dict(reg_class_agnostic=True),
    ),
    train_cfg=dict(
        rpn_proposal=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            _delete_=True,
            assigner=dict(
                type='SimOTAAssigner',
                center_radius=2.5,
                candidate_topk=10,
                iou_weight=3.0,
                cls_weight=1.0),
            # SimOTA already produces final 1-to-1 pos/neg; PseudoSampler
            # forwards them untouched. Avoid RandomSampler with
            # add_gt_as_proposals=True here (would change the assignment).
            sampler=dict(type='PseudoSampler'),
            pos_weight=-1,
            debug=False),
    ),
)
