"""Ablation: NWD-RKA + soft-label on NWD (instead of IoU) in the RPN.

Same as ``nwdrka_softrpn.py`` (the faithful IoU-soft version) but the soft
objectness TARGET is the area-refine calibrated NWD quality instead of IoU.
Matching is NWD in both. This isolates the effect of the soft-label METRIC:
- nwdrka_softrpn.py      : NWD matching + IoU soft label (original USAA intent)
- nwdrka_softrpn_nwd.py  : NWD matching + NWD soft label (scale-invariant target)
"""

_base_ = ['./_base_nwdrka_softrpn.py']

model = dict(rpn_head=dict(quality_metric='nwd'))
