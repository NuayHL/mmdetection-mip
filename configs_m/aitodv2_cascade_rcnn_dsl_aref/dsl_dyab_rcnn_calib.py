"""DSL-DYAB-RCNN with a CALIBRATED soft-label supervision signal (Cascade R-CNN).

Identical to ``dsl_dyab_rcnn_default.py`` (3-stage DSL-DScale-DYAB on Cascade
R-CNN) EXCEPT the assigner type in every stage:

    DynamicSoftLabelAssignerDScaleDYAB      → calibration feeds only the
                                              matching cost; the delivered soft
                                              label is the RAW matched IoU.
    DynamicSoftLabelAssignerDScaleDYABCalib → same matching, but the soft label
                                              written to max_overlaps (→ QFL via
                                              DynAssignCascadeRoIHead) is
                                              area-refine calibrated.

mmengine REPLACES list elements (no deep-merge), so all three stages are
restated in full; only ``assigner.type`` differs from the default config.
Pair with ``dsl_dyab_rcnn_default.py`` for the A/B isolating the calibrated
supervision signal.
"""

_base_ = ['./dsl_dyab_rcnn_default.py']

model = dict(
    train_cfg=dict(
        rcnn=[
            dict(
                _delete_=True,
                assigner=dict(
                    type='DynamicSoftLabelAssignerDScaleDYABCalib',
                    soft_center_radius=2.0,
                    topk=13,
                    iou_weight=3.0,
                    r_ref=32.0,
                    calibrate_mode='add_1',
                    scale_ratio=1.0,
                    expansion_type='static',
                    expansion_r_ref=32.0,
                    dyab_type='DyabDSL',
                    dyab_kwargs=dict(
                        alpha_base=0.8,
                        beta_base=1.5,
                        delta_alpha=0.4,
                        delta_beta=0.5,
                        r_ref=64.0,
                    ),
                ),
                sampler=dict(type='PseudoSampler'),
                pos_weight=-1,
                debug=False,
            ),
            dict(
                _delete_=True,
                assigner=dict(
                    type='DynamicSoftLabelAssignerDScaleDYABCalib',
                    soft_center_radius=2.0,
                    topk=13,
                    iou_weight=3.0,
                    r_ref=32.0,
                    calibrate_mode='add_1',
                    scale_ratio=1.0,
                    expansion_type='static',
                    expansion_r_ref=32.0,
                    dyab_type='DyabDSL',
                    dyab_kwargs=dict(
                        alpha_base=0.8,
                        beta_base=1.5,
                        delta_alpha=0.4,
                        delta_beta=0.5,
                        r_ref=64.0,
                    ),
                ),
                sampler=dict(type='PseudoSampler'),
                pos_weight=-1,
                debug=False,
            ),
            dict(
                _delete_=True,
                assigner=dict(
                    type='DynamicSoftLabelAssignerDScaleDYABCalib',
                    soft_center_radius=2.0,
                    topk=13,
                    iou_weight=3.0,
                    r_ref=32.0,
                    calibrate_mode='add_1',
                    scale_ratio=1.0,
                    expansion_type='static',
                    expansion_r_ref=32.0,
                    dyab_type='DyabDSL',
                    dyab_kwargs=dict(
                        alpha_base=0.8,
                        beta_base=1.5,
                        delta_alpha=0.4,
                        delta_beta=0.5,
                        r_ref=64.0,
                    ),
                ),
                sampler=dict(type='PseudoSampler'),
                pos_weight=-1,
                debug=False,
            ),
        ],
    ),
)
