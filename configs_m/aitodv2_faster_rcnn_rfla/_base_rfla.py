"""RFLA (Receptive Field based Label Assignment) base config for AITOD-v2
Faster R-CNN experiments.

Key differences from the MaxIoU baseline:
  * Anchor generator: RFGenerator — uses theoretical receptive field
    (TRF) of ResNet-50-FPN instead of fixed stride-based base sizes.
  * RPN assigner: HieAssigner — two-stage hierarchical label assignment
    with KLD (KL divergence between 2D Gaussians) as the affinity metric.

All other settings (dataset, schedule, RPN proposals, RoI head, test)
are inherited from the ``_base_.py`` baseline.
"""

_base_ = ['../aitodv2_faster_rcnn/_base_.py']

# Override to use RFGenerator and HieAssigner for RFLA
model = dict(
    rpn_head=dict(
        # Replace standard AnchorGenerator with RFGenerator.
        # For two-stage detectors, the FPN base layer is P2.
        anchor_generator=dict(
            type='RFGenerator',
            fpn_layer='p2',
            fraction=0.5,
            strides=[4, 8, 16, 32, 64]),
    ),
    train_cfg=dict(
        rpn=dict(
            # Replace MaxIoUAssigner with HieAssigner for RFLA
            assigner=dict(
                _delete_=True,
                type='HieAssigner',
                ignore_iof_thr=-1,
                gpu_assign_thr=512,
                iou_calculator=dict(type='BboxDistanceMetric'),
                assign_metric='kl',
                topk=[3, 1],
                ratio=0.9),
        ),
    ),
)
