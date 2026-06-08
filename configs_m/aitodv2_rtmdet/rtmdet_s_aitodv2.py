"""RTMDet-s baseline on AITOD-v2.

Standard RTMDet recipe (AdamW + cosine, CachedMosaic/MixUp, EMA, two-stage
pipeline switch), adapted to AITOD-v2:

  * RTMDet-s backbone/neck/head (deepen=0.33, widen=0.5), num_classes=8
  * input resolution 800x800 (AITOD images are 800x800; tiny objects need it)
  * 100 epochs, last 10 epochs use the plain stage-2 pipeline
    (~11.2k train imgs / bs 16 -> ~700 iters/epoch -> ~70k iters total)
  * default ``DynamicSoftLabelAssigner`` (the RTMDet baseline assigner)

Self-contained: the model is defined inline (this repo has no
``configs/_base_`` and no RTMDet model base under ``configs_m/_base_``), so
only ``default_runtime`` is inherited.  This file also serves as the shared
base for the USAA-assigner experiment ``rtmdet_s_aitodv2_usaa.py``, which
overrides only ``train_cfg.assigner``.

Note: the model uses ``SyncBN`` (standard RTMDet).  For single-GPU,
non-distributed training switch ``norm_cfg`` to ``dict(type='BN')``.
"""

_base_ = ['../_base_/default_runtime.py']

# ── AITOD-v2 dataset ────────────────────────────────────────────────────────
dataset_type = 'CocoDataset'
data_root = 'data/ai-todv2/'
backend_args = None

aitod_classes = (
    'airplane', 'bridge', 'storage-tank', 'ship',
    'swimming-pool', 'vehicle', 'person', 'wind-mill',
)

# ── schedule / resolution ───────────────────────────────────────────────────
max_epochs = 100
stage2_num_epochs = 10
interval = 10
img_scale = (800, 800)

# total batch = train_batch_size_per_gpu x #GPUs.  base_lr is linearly scaled
# with the total batch (0.0005 @ bs 8 -> 0.001 @ bs 16).  If bs 16 OOMs at
# 800px, drop to 12 (lr 0.00075) or 8 (lr 0.0005), or launch with
# ``--auto-scale-lr`` for multi-GPU runs.
train_batch_size_per_gpu = 16
train_num_workers = 8
base_lr = 0.001

# ── model: RTMDet-s ─────────────────────────────────────────────────────────
checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth'  # noqa
norm_cfg = dict(type='SyncBN')

model = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        batch_augments=None),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.33,
        widen_factor=0.5,
        channel_attention=True,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True),
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1,
        expand_ratio=0.5,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='RTMDetSepBNHead',
        num_classes=8,
        in_channels=128,
        stacked_convs=2,
        feat_channels=128,
        anchor_generator=dict(
            type='MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=False,
        share_conv=True,
        pred_kernel_size=1,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=30000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300),
)

# ── pipelines (800x800) ─────────────────────────────────────────────────────
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CachedMosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomResize',
        scale=(img_scale[0] * 2, img_scale[1] * 2),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=img_scale),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='CachedMixUp',
        img_scale=img_scale,
        ratio_range=(1.0, 1.0),
        max_cached_images=20,
        pad_val=(114, 114, 114)),
    dict(type='PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=img_scale,
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=img_scale),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# ── dataloaders ─────────────────────────────────────────────────────────────
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
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
        data_prefix=dict(img='images/val/'),
        metainfo=dict(classes=aitod_classes),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = val_dataloader

# AI-TOD is very dense (~27 objects/image), so the COCO metric must allow many
# detections per image.  The headline AP/AP50/AP75 use proposal_nums[2] as
# maxDets, hence 1500 here (the AI-TOD standard) — NOT COCO's 10/100.  With the
# old (100, 1, 10) the AP was computed at only 10 dets/image and was wildly
# under-reported vs. the standalone eval_aitod.py (which uses 1500).
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations_v2/aitodv2_val_fixed.json',
    metric='bbox',
    format_only=False,
    proposal_nums=(1, 100, 1500),
    backend_args=backend_args,
    classwise=True)
test_evaluator = val_evaluator

# ── train / optim / schedule (100 epochs) ───────────────────────────────────
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=interval,
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0,
         end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

auto_scale_lr = dict(enable=False, base_batch_size=256)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=interval,
                    max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=100))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]
