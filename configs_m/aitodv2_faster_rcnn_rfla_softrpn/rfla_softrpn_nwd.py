"""Ablation: RFLA (KLD matching) + soft-label on NWD instead of IoU.

Same as ``rfla_softrpn.py`` (KLD matching + IoU soft label) but the soft
objectness TARGET is the area-refine calibrated NWD quality instead of IoU.
Matching stays KLD in both. This isolates the effect of the soft-label
METRIC under the RFLA assigner:
- rfla_softrpn.py      : KLD matching + IoU soft label (original USAA intent)
- rfla_softrpn_nwd.py  : KLD matching + NWD soft label (scale-invariant target)
"""

_base_ = ['./_base_rfla_softrpn.py']

model = dict(rpn_head=dict(quality_metric='nwd'))
