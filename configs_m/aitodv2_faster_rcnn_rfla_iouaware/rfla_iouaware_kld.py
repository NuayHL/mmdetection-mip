"""RFLA-KLD (RPN) + decoupled IoU-aware RCNN head.

RPN:  HieAssigner with KLD metric (RFLA default), unchanged.
RCNN: IoU-aware head, score = softmax_cls * sigmoid(iou_pred), area-refine
      calibrated IoU target.

Compare against configs_m/aitodv2_faster_rcnn_rfla/aitodv2_rfla_kld.py (0.216).
"""

_base_ = ['./_base_rfla_iouaware.py']
