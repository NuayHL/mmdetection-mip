"""RFLA + USAA soft-label fused in the RPN (primary fusion experiment).

RPN:   HieAssigner (KLD top-k, two-stage) + SoftLabelRPNHead (soft IoU
       objectness, QFL). Matching = KLD, soft label = IoU.
RCNN:  RFLA baseline (StandardRoIHead + MaxIoU + CE, pos_fraction=0.3).

Goal: beat the plain RFLA-KLD baseline
(configs_m/aitodv2_faster_rcnn_rfla/aitodv2_rfla_kld.py). If so, the paper
story "we improve the SOTA tiny-object RPN assigner with our soft-label
mechanism" holds for RFLA as well as NWD-RKA.
"""

_base_ = ['./_base_rfla_softrpn.py']
