"""Shared base for the AITOD-v2 Detectors DSL-AreaRefine experiments.

Built on the DetectoRS_ResNet + RFP backbone with the 3-stage cascade
refinement mechanism.  The roi_head and train_cfg.rcnn are overridden
per-experiment to use ``DynAssignCascadeRoIHead`` with DSL assigners.

Holds every setting shared across experiments:
  * dataset / annotation paths / pipelines
  * training schedule (24 epochs, SGD, multistep)
  * model: DetectoRS backbone / RFP neck / RPN head / RPN proposal count (3000)
  * test-time NMS / top-k (3000)

Per-experiment configs override ``model.roi_head`` and
``model.train_cfg.rcnn`` (as a list of 3 stage configs).
"""

_base_ = [
    '../_base_/models/cascade-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

aitod_classes = (
    'airplane', 'bridge', 'storage-tank', 'ship',
    'swimming-pool', 'vehicle', 'person', 'wind-mill',
)

max_epochs = 24
train_batch_size_per_gpu = 2
train_num_workers = 4
save_epoch_intervals = 1

# Keep CascadeRCNN type + add DetectoRS+RFP backbone/neck.
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
    roi_head=dict(
        bbox_head=[
            dict(num_classes=8),
            dict(num_classes=8),
            dict(num_classes=8),
        ],
    ),
    train_cfg=dict(
        rpn=dict(
            sampler=dict(pos_fraction=0.3),
        ),
        rpn_proposal=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=3000,
            max_per_img=3000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=3000),
    ))

dataset_type = 'CocoDataset'
data_root = 'data/ai-todv2/'
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations_v2/aitodv2_train_fixed.json',
        data_prefix=dict(img='images/train/'),
        metainfo=dict(classes=aitod_classes),
        filter_cfg=dict(filter_empty_gt=False, min_size=4),
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
        data_root=data_root,
        ann_file='annotations_v2/aitodv2_val_fixed.json',
        metainfo=dict(classes=aitod_classes),
        data_prefix=dict(img='images/val/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations_v2/aitodv2_val_fixed.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args,
    classwise=True)
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
         end=500),
    dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True,
         milestones=[16, 22], gamma=0.1),
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001))

auto_scale_lr = dict(enable=False, base_batch_size=train_batch_size_per_gpu)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=save_epoch_intervals),
    logger=dict(type='LoggerHook', interval=100))
