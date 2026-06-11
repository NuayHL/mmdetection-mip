"""Ablation: dampened IoU fusion (alpha=0.5).

score = softmax_cls * sigmoid(iou_pred) ** 0.5. Tests whether softening the
IoU weighting helps vs the faithful alpha=1.
"""

_base_ = ['./_base_nwdrka_iouaware.py']

model = dict(roi_head=dict(bbox_head=dict(iou_alpha=0.5)))
