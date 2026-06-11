"""RFLA-WD (RPN) + decoupled IoU-aware RCNN head.

Same as rfla_iouaware_kld.py but the HieAssigner uses the Wasserstein
distance (assign_metric='wd') for RPN matching/selection. RCNN IoU-aware head
unchanged.

Compare against configs_m/aitodv2_faster_rcnn_rfla/aitodv2_rfla_wd.py (0.222).
"""

_base_ = ['./_base_rfla_iouaware.py']

model = dict(
    train_cfg=dict(
        rpn=dict(
            assigner=dict(assign_metric='wd'),
        ),
    ),
)
