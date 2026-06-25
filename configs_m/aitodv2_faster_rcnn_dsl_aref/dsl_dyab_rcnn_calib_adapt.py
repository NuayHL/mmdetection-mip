"""DSL-DYAB-RCNN with a CALIBRATED soft-label supervision signal.

Byte-for-byte identical to ``dsl_dyab_rcnn_default.py`` (the DSL-DScale-DYAB
RCNN config that already gives large gains on the plain Faster R-CNN baseline)
EXCEPT the assigner type:

    DynamicSoftLabelAssignerDScaleDYAB      → matching uses calibrated IoU, but
                                              the soft label delivered to the
                                              loss is the RAW matched IoU.
    DynamicSoftLabelAssignerDScaleDYABCalib → same matching, but the delivered
                                              soft label is area-refine
                                              calibrated (calibrate_mode='add_1',
                                              r_ref=32, inherited).

This makes the "my soft label is calibrated" claim honest on the two-stage
baseline — the supervision signal, not just the matching cost, now carries the
calibration. Pair with ``dsl_dyab_rcnn_default.py`` for the A/B that isolates
the calibrated-supervision effect (or set ``calibrate_target=False`` on this
assigner to recover the parent exactly).
"""

_base_ = ['./dsl_dyab_rcnn_adapt.py']

model = dict(
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(type='DynamicSoftLabelAssignerDScaleDYABCalib'),
        ),
    ),
)