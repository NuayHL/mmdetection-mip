"""Ablation: raw IoU target, no area-refine calibration (alpha=1).

The IoU branch regresses the raw IoU(decoded_pred, gt) without lifting the
small-object ceiling. Isolates the contribution of the area-refine
calibration (your core USAA idea) vs a plain IoU-aware head.
"""

_base_ = ['./_base_nwdrka_iouaware.py']

model = dict(roi_head=dict(bbox_head=dict(calibrate=False)))
