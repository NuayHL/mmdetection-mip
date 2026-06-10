"""RFLA x USAA fusion at the RPN (the right stage).

Same story as the NWD-RKA softrpn fusion, but with RFLA's hierarchical
assigner as the matching/selection mechanism instead of NWD-RKA's
RankingAssigner. Tiny-object gains live at the RPN assignment stage, so we
move USAA's soft-label idea to the RPN and FUSE it with RFLA along
orthogonal axes:

  * RFLA contributes the METRIC + selection: RFGenerator anchors (P2-based
    TRF prior) + HieAssigner, which ranks anchors by KLD (KL divergence
    between 2D Gaussians) and assigns the top-k per GT as positives in two
    hierarchical stages (unchanged). The original RFLA then supervises
    objectness with a HARD 1/0 CrossEntropy label.
  * USAA contributes the SUPERVISION: SoftLabelRPNHead turns that hard 1
    into a *soft* area-refine calibrated quality, trained with
    QualityFocalLoss (GFL-style). Matching (KLD, in the assigner) and the
    soft-label metric (in the head) are independent axes: matching stays
    KLD, while the soft target defaults to IoU (``quality_metric='iou'``) —
    faithful to the original USAA/YOLO design where the soft label is IoU
    quality. ``quality_metric='nwd'`` is a separate ablation that makes the
    soft target scale-invariant too.

The RCNN stage is left EXACTLY as the RFLA baseline (StandardRoIHead +
MaxIoU + CE, pos_fraction=0.3) so this isolates the RPN fusion. Compare
directly against configs_m/aitodv2_faster_rcnn_rfla/aitodv2_rfla_kld.py.
"""

_base_ = ['../aitodv2_faster_rcnn_rfla/_base_rfla.py']

model = dict(
    rpn_head=dict(
        type='SoftLabelRPNHead',
        # soft objectness = area-refine calibrated IoU(decoded_pred, gt).
        # Matching is still KLD (HieAssigner); only the soft TARGET is IoU,
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
    train_cfg=dict(
        rcnn=dict(
            sampler=dict(pos_fraction=0.3),
        ),
    ),
)
