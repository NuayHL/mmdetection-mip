"""Ablation: 'pow' area-refine calibration mode (alpha=1).

Uses calibrate_mode='pow' (q ** rho) instead of the default 'add_1' for the
IoU target. Compares the two area-refine calibration formulas.
"""

_base_ = ['./_base_nwdrka_iouaware.py']

model = dict(roi_head=dict(bbox_head=dict(calibrate_mode='pow')))
