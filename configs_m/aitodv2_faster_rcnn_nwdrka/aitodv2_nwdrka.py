"""NWD-RKA Faster R-CNN on AITOD-v2.

RPN positive/negative assignment uses RankingAssigner with the Normalized
Wasserstein Distance metric (top-2 proposals per GT). Everything else
follows the shared AITOD-v2 Faster R-CNN baseline.
"""

_base_ = ['./_base_nwdrka.py']
