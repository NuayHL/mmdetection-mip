"""RFLA-KLD RPN + CALIBRATED DSL-DYAB RCNN  (Faster R-CNN).

The "modify a little to combine with RFLA" experiment for the calibrated
two-stage soft-label method. Inherits ``dsl_dyab_rcnn_calib.py`` (baseline
Faster R-CNN + DynAssignRoIHead + DynamicSoftLabelAssignerDScaleDYABCalib, the
version that beats the non-calibrated default by ~2 AP50) and changes ONLY the
RPN:

  * AnchorGenerator → RFGenerator (P2-based TRF priors)
  * RPN MaxIoUAssigner → HieAssigner (KLD metric, topk=[3,1], ratio=0.9)

The RCNN stage (calibrated soft label) is untouched. Earlier the
*non*-calibrated DSL-DYAB did not improve when combined with RFLA; this re-tests
whether the calibrated-supervision variant behaves differently. Compare against
``dsl_dyab_rcnn_calib.py`` (no RFLA) and against the RFLA baseline.
"""

_base_ = ['./dsl_dyab_rcnn_calib.py']

model = dict(
    rpn_head=dict(
        anchor_generator=dict(
            type='RFGenerator',
            fpn_layer='p2',
            fraction=0.5,
            strides=[4, 8, 16, 32, 64]),
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                _delete_=True,
                type='HieAssigner',
                ignore_iof_thr=-1,
                gpu_assign_thr=512,
                iou_calculator=dict(type='BboxDistanceMetric'),
                assign_metric='kl',
                topk=[3, 1],
                ratio=0.9),
        ),
    ),
)