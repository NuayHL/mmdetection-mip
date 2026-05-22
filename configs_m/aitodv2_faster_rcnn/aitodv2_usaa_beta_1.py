"""USAA — ``QualityFocalLoss.beta = 1.0`` (default 2.0).

Lower beta → milder focal modulation, easy negatives contribute more.
"""

_base_ = ['./aitodv2_usaa.py']

model = dict(
    roi_head=dict(
        bbox_head=dict(
            loss_cls=dict(beta=1.0),
        ),
    ),
)
