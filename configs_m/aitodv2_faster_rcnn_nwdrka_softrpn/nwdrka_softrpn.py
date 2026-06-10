"""NWD-RKA + USAA soft-label fused in the RPN (primary fusion experiment).

RPN:   RankingAssigner (NWD top-k) + SoftLabelRPNHead (soft NWD objectness, QFL)
RCNN:  NWD-RKA baseline (StandardRoIHead + MaxIoU + CE), unchanged.

Goal: beat the plain NWD-RKA baseline (mAP 0.216) — if so, the paper story
"we improve the SOTA tiny-object RPN assigner with our soft-label mechanism"
holds.
"""

_base_ = ['./_base_nwdrka_softrpn.py']
