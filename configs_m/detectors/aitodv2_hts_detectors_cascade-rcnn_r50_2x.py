_base_ = [
    './aitodv2_iou_detectors_cascade-rcnn_r50_2x.py',
]

model = dict(
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxSiMAssigner',
                pos_iou_thr=0.95,
                neg_iou_thr=0.5,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='BboxSiM2D',
                                    mode='hausdorff',
                                    iou_kwargs=dict(lambda1=2.5,
                                                    lambda3=7,
                                                    using_central=True,
                                                    pow_value=4.0))),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.3,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=3000,
            max_per_img=3000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.3,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    )
