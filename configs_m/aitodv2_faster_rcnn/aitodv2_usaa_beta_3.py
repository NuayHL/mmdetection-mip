"""USAA — ``QualityFocalLoss.beta = 3.0`` (default 2.0).

Higher beta → sharper focal modulation, focus more on hard examples.
"""

_base_ = ['./aitodv2_usaa.py']

model = dict(
    roi_head=dict(
        bbox_head=dict(
            loss_cls=dict(beta=3.0),
        ),
    ),
)
