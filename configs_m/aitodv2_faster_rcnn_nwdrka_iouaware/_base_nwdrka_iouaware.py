"""NWD-RKA (RPN) + decoupled IoU-aware RCNN head (USAA, done right).

The soft-IoU idea, placed so it does NOT fight NWD-RKA and does NOT lower the
ranking score of true positives:

  * RPN  : NWD-RKA unchanged (RankingAssigner, NWD top-k). Hard objectness.
  * RCNN : IoUAwareRoIHead + IoUAwareShared2FCBBoxHead.
           - The ORIGINAL softmax + CrossEntropy classification head is kept
             byte-for-byte (num_classes + 1 channels, F.softmax). So the
             ranking score of a true positive is never lowered, and the head
             stays structurally comparable to the baseline.
           - A SEPARATE scalar IoU branch predicts the area-refine calibrated
             IoU(decoded_pred, gt) of each positive RoI (sigmoid BCE).
           - At inference: score = softmax_cls * sigmoid(iou_pred) ** alpha.
             The soft IoU signal is ADDED to the ranking score (which the
             baseline lacks entirely), not used to replace the cls target.

This is the fix for the failure shared by softrpn / softiou: both put the soft
target on the very logit used for ranking, suppressing recall. Here the cls
logit is untouched and the soft signal lives on an independent branch.

Compare against configs_m/aitodv2_faster_rcnn_nwdrka/aitodv2_nwdrka.py (0.216).
"""

_base_ = ['../aitodv2_faster_rcnn_nwdrka/_base_nwdrka.py']

model = dict(
    roi_head=dict(
        type='IoUAwareRoIHead',
        bbox_head=dict(
            type='IoUAwareShared2FCBBoxHead',
            num_classes=8,
            # decoupled IoU branch (USAA area-refine on localization quality).
            # alpha=1.0 is the faithful soft label: score = cls * sigmoid(iou).
            iou_alpha=1.0,
            r_ref=32.0,
            calibrate_mode='add_1',
            calibrate=True,
            loss_iou_weight=1.0,
            # classification path stays EXACTLY the baseline (softmax + CE)
        ),
    ),
)
