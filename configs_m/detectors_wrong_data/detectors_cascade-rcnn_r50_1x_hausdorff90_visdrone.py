_base_ = [
    '../_base_/models/cascade-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

max_epochs = 12  # 训练的最大 epoch
work_dir = './work_dirs/visdrone/faster-rcnn_r50_fpn_1x_iou_visdrone'
train_batch_size_per_gpu = 2
train_num_workers = 4  # 推荐使用 train_num_workers = nGPU x 4
save_epoch_intervals = 1  # 每 interval 轮迭代进行一次保存一次权重

# val_json = 'visdrone_val_fix_fixed.json'
val_json = 'visdrone_test_coco_format.json'
# val_main_folder = 'VisDrone2019-DET-val/'
val_main_folder = 'VisDrone2019-DET-test-val/'

model = dict(
    backbone=dict(
        type='DetectoRS_ResNet',
        conv_cfg=dict(type='ConvAWS'),
        sac=dict(type='SAC', use_deform=True),
        stage_with_sac=(False, True, True, True),
        output_img=True),
    neck=dict(
        type='RFP',
        rfp_steps=2,
        aspp_out_channels=64,
        aspp_dilations=(1, 3, 6, 1),
        rfp_backbone=dict(
            rfp_inplanes=256,
            type='DetectoRS_ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            conv_cfg=dict(type='ConvAWS'),
            sac=dict(type='SAC', use_deform=True),
            stage_with_sac=(False, True, True, True),
            pretrained='torchvision://resnet50',
            style='pytorch')),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                type='MaxSiMAssigner',
                pos_iou_thr=0.9,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='BboxSiM2D',
                                    mode='hausdorff',
                                    iou_kwargs=dict())),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                type='MaxSiMAssigner',
                pos_iou_thr=0.93,
                neg_iou_thr=0.8,
                min_pos_iou=0.8,
                match_low_quality=True,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='BboxSiM2D',
                                    mode='hausdorff',
                                    iou_kwargs=dict())),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                type='MaxSiMAssigner',
                pos_iou_thr=0.95,
                neg_iou_thr=0.85,
                min_pos_iou=0.85,
                match_low_quality=True,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='BboxSiM2D',
                                    mode='hausdorff',
                                    iou_kwargs=dict())),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),)

dataset_type = 'CocoDataset'
data_root = 'data/VisDrone/'

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root + 'VisDrone2019-DET-train/',
        ann_file='../visdrone_train_fix_fixed.json',
        data_prefix=dict(img='images/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root + val_main_folder,
        ann_file=f'../{val_json}',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='VisDroneCocoMetric',
    ann_file=data_root + val_json,
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

test_evaluator = val_evaluator

# training schedule for 1x
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001))

auto_scale_lr = dict(enable=False, base_batch_size=train_batch_size_per_gpu)

default_hooks = dict(
    # 设置间隔多少个 epoch 保存模型，以及保存模型最多几个，`save_best` 是另外保存最佳模型（推荐）
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_epoch_intervals),
    # param_scheduler=dict(max_epochs=max_epochs),
    # logger 输出的间隔
    logger=dict(type='LoggerHook', interval=100))
