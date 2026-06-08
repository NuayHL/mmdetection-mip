"""RTMDet-s on AITOD-v2 with the USAA task-aligned assigner.

Identical to the baseline ``rtmdet_s_aitodv2.py`` in every respect except the
training assigner, which is swapped from the default ``DynamicSoftLabelAssigner``
to ``TaskAlignedAssignerDScaleDYAB`` — the one-stage port of the ultralytics
``TaskAlignedAssigner_dyab_dmetric_dscale_RefineArea``.

The assigner parameters below mirror the ultralytics defaults exactly:
  alignment metric : s^alpha * IoU^beta, dynamic (alpha, beta) via DyabCalibrationAware
  soft label       : area-refine ceiling calibration (add_1), r_ref=32
  dscale           : static candidate region, scale_ratio=1.0
  IoU              : CIoU for all roles

Note: ``alpha``/``beta`` in the ultralytics config are fallbacks that the dyab
strategy overrides, so they are not passed here — ``DyabCalibrationAware``
fully determines (alpha, beta).
"""

_base_ = ['./rtmdet_s_aitodv2.py']

model = dict(
    train_cfg=dict(
        assigner=dict(
            _delete_=True,
            type='TaskAlignedAssignerDScaleDYAB',
            topk=10,
            iou_calculator=dict(type='BboxSiM2D', mode='ciou'),
            dscale_func='static',
            scale_ratio=1.0,
            r_ref=32.0,
            r_ref_type='add_1',
            r_ref_use_adaptive=False,
            dyab_type='DyabCalibrationAware',
            dyab_kwargs=dict(
                alpha_base=1.0,
                beta_base=4.0,
                delta_alpha=0.5,
                delta_beta=2.0,
                r_ref=64.0,
            ),
        ),
    ),
)
