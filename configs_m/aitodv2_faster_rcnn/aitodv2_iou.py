"""Baseline: standard Faster R-CNN with MaxIoUAssigner + RandomSampler.

This config matches the previous ``aitodv2_iou_r50_fpn_2x.py`` exactly,
but is rebuilt on top of ``_base_.py`` so it inherits the identical
dataset, schedule, RPN, and RPN-proposal pool (3000) used by every USAA
experiment in this folder.

The only change versus the base FasterRCNN config (``faster-rcnn_r50_fpn.py``)
in the RCNN stage is the sampler's ``pos_fraction = 0.3`` (default 0.25);
the assigner stays as MaxIoUAssigner(pos_iou_thr=0.5, ...).
"""

_base_ = ['./_base_.py']

model = dict(
    train_cfg=dict(
        rcnn=dict(
            sampler=dict(pos_fraction=0.3),
        ),
    ),
)
