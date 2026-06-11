"""NWD-RKA (RPN) + decoupled IoU-aware RCNN head (primary experiment).

score = softmax_cls * sigmoid(iou_pred) (alpha=1, the faithful soft IoU
label), area-refine calibrated IoU target. Goal: beat the NWD-RKA baseline
(mAP 0.216) by adding localization quality to the ranking score WITHOUT
lowering the cls score of true positives.
"""

_base_ = ['./_base_nwdrka_iouaware.py']
