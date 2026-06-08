"""RFLA-KLD: Receptive Field based Label Assignment with KL Divergence.

Standard RFLA configuration from the paper:
  * RFGenerator anchors (P2-based, TRF as prior)
  * HieAssigner with KLD affinity metric
  * Two-stage assignment: topk=[3, 1], ratio=0.9
  * RCNN stage uses standard MaxIoUAssigner (unchanged from baseline)

Compared to the MaxIoU baseline, only the RPN anchor generator and
assigner differ — everything else (dataset, schedule, RoI head,
proposal pool size) is identical for a fair comparison.
"""

_base_ = ['./_base_rfla.py']

model = dict(
    train_cfg=dict(
        rcnn=dict(
            sampler=dict(pos_fraction=0.3),
        ),
    ),
)
