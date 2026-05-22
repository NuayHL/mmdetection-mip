"""USAA — default hyperparameters.

Prediction-aware label assignment in the R-CNN stage:
  * DynAssignRoIHead with ``prior_format='point'`` converts each RPN
    proposal to (cx, cy, min(w,h), min(w,h)) before the assigner sees it,
    matching the point-prior layout DynamicSoftLabelAssigner expects.
  * DynamicSoftLabelAssigner picks positives by an (IoU + soft-label cls)
    cost; ``max_overlaps`` carries the per-pos IoU.
  * QualityFocalLoss with ``custom_cls_channels=True`` consumes that IoU
    as a soft cls target and emits sigmoid scores at inference (no dead
    bg channel).
  * PseudoSampler keeps every assigned positive for the loss; the entire
    RPN-proposal pool (3000) is kept consistent with the IoU baseline.
"""

_base_ = ['./_base_.py']

model = dict(
    roi_head=dict(
        type='DynAssignRoIHead',
        cls_score_activation='identity',
        prior_format='point',
        use_iou_soft_target=True,
        bbox_head=dict(
            reg_class_agnostic=True,
            loss_cls=dict(
                type='QualityFocalLoss',
                use_sigmoid=True,
                beta=2.0,
                loss_weight=1.0,
                custom_cls_channels=True),
        ),
    ),
    train_cfg=dict(
        rcnn=dict(
            _delete_=True,
            assigner=dict(
                type='DynamicSoftLabelAssigner',
                soft_center_radius=3.0,
                topk=13,
                iou_weight=3.0),
            sampler=dict(type='PseudoSampler'),
            pos_weight=-1,
            debug=False),
    ),
)
