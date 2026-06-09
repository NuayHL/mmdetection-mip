"""NWD-RKA (RPN) + MaxSoftIoU (RCNN).

RPN:   NWD-RKA (RankingAssigner with NWD metric, topk=2)
RCNN:  MaxSoftIoUAssigner (area-refine soft-label IoU) + QFL soft target.

The lightweight alternative to nwdrka_dsl_dyab: keeps the soft label, drops
the dynamic-assignment machinery.
"""

_base_ = ['./_base_nwdrka_softiou.py']
