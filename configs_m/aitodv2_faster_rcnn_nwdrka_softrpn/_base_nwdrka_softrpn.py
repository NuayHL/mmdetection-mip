"""NWD-RKA x USAA fusion at the RPN (the right stage).

Story: tiny-object gains live at the RPN assignment stage (NWD-RKA gets
AP_s 0.124->0.233 purely from the RPN; USAA at the RCNN gets only +0.03).
So we move USAA's soft-label idea to the RPN and FUSE it with NWD-RKA along
orthogonal axes:

  * NWD-RKA contributes the METRIC + selection: RankingAssigner ranks anchors
    by NWD and assigns the top-k per GT as positives (unchanged). The original
    NWD-RKA then supervises objectness with a HARD 1/0 CrossEntropy label.
  * USAA contributes the SUPERVISION: SoftLabelRPNHead turns that hard 1 into
    a *soft* area-refine calibrated quality, trained with QualityFocalLoss
    (GFL-style). Matching (NWD) and the soft-label metric are independent
    axes: matching stays NWD (the assigner), while the soft target defaults to
    IoU (``quality_metric='iou'``) — faithful to the original USAA/YOLO design
    where the soft label is IoU quality. ``quality_metric='nwd'`` is a separate
    ablation that makes the soft target scale-invariant too.

The RCNN stage is left EXACTLY as the NWD-RKA baseline (StandardRoIHead +
MaxIoU + CE) so this isolates the RPN fusion. Compare directly against
configs_m/aitodv2_faster_rcnn_nwdrka/aitodv2_nwdrka.py (mAP 0.216).
"""

_base_ = ['../aitodv2_faster_rcnn_nwdrka/_base_nwdrka.py']

model = dict(
    rpn_head=dict(
        type='SoftLabelRPNHead',
        # soft objectness = area-refine calibrated IoU(decoded_pred, gt).
        # Matching is still NWD (RankingAssigner); only the soft TARGET is IoU,
        # which is faithful to the original USAA/YOLO soft-label design.
        quality_metric='iou',
        constant=12.7,
        r_ref=32.0,
        calibrate_mode='add_1',
        soft_label=True,
        loss_cls=dict(
            _delete_=True,
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
    ),
)
