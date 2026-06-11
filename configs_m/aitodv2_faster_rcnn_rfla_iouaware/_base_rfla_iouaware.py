"""RFLA (RPN) + decoupled IoU-aware RCNN head (USAA, done right).

The RFLA counterpart of the NWD-RKA IoU-aware fusion. The soft-IoU idea is
placed so it does NOT fight RFLA's assigner and does NOT lower the ranking
score of true positives:

  * RPN  : RFLA unchanged (RFGenerator anchors + HieAssigner, KLD/WD top-k).
           Hard objectness.
  * RCNN : IoUAwareRoIHead + IoUAwareShared2FCBBoxHead.
           - Original softmax + CrossEntropy classification head kept
             byte-for-byte (num_classes + 1 channels, F.softmax).
           - A SEPARATE scalar IoU branch predicts the area-refine calibrated
             IoU(decoded_pred, gt) of each positive RoI (sigmoid BCE).
           - At inference: score = softmax_cls * sigmoid(iou_pred) ** alpha,
             alpha=1 (faithful soft IoU). The soft signal is ADDED to the
             ranking score, not used to replace the cls target.

RCNN sampler keeps pos_fraction=0.3 to match the RFLA baselines (rfla_kld /
rfla_wd) for a fair comparison.
"""

_base_ = ['../aitodv2_faster_rcnn_rfla/_base_rfla.py']

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
    train_cfg=dict(
        rcnn=dict(
            sampler=dict(pos_fraction=0.3),
        ),
    ),
)
