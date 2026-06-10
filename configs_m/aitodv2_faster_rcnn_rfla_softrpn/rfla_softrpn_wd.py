"""Ablation: RFLA-WD matching + USAA soft IoU label in the RPN.

Same as ``rfla_softrpn.py`` but the HieAssigner uses the Wasserstein
distance (``assign_metric='wd'``) instead of KLD for matching/selection.
The soft objectness TARGET is still IoU. This pairs with
configs_m/aitodv2_faster_rcnn_rfla/aitodv2_rfla_wd.py as the WD baseline.
"""

_base_ = ['./_base_rfla_softrpn.py']

model = dict(
    train_cfg=dict(
        rpn=dict(
            assigner=dict(assign_metric='wd'),
        ),
    ),
)
