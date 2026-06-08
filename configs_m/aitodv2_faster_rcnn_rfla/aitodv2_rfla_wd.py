"""RFLA-WD: Receptive Field based Label Assignment with Wasserstein Distance.

Variant of RFLA that uses Wasserstein distance instead of KL divergence
as the affinity metric between 2D Gaussian bounding box distributions.
"""

_base_ = ['./_base_rfla.py']

model = dict(
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                assign_metric='wd',
            ),
        ),
        rcnn=dict(
            sampler=dict(pos_fraction=0.3),
        ),
    ),
)
