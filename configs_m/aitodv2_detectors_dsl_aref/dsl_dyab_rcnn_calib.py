"""DSL-DYAB-RCNN with a CALIBRATED soft-label supervision signal (Detectors).

Identical to ``dsl_dyab_rcnn_default.py`` (3-stage DSL-DScale-DYAB on the
DetectoRS+RFP backbone) EXCEPT the assigner type in every stage:

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

# Gradient clipping — the DSL-DScale-DYAB base configs ship WITHOUT it. The
# calibrated soft label raises small-object cls confidence → the prediction-
# aware DSL matching picks more tiny-box positives → more regression stress on
# tiny/degenerate boxes, which the 3-stage cascade refinement amplifies into a
# single-step bbox blow-up at ~epoch 14 (s1/s2.loss_bbox → inf → loss nan →
# dead weights). Single-stage Faster R-CNN has no cross-stage amplification, so
# it stayed stable without clipping. max_norm=35 is the mmdet standard; if it
# still diverges, drop to 10 and/or lower lr (0.005→0.0035). For a clean A/B,
# add the same clip to the non-calibrated default before comparing.
optim_wrapper = dict(clip_grad=dict(max_norm=35, norm_type=2))

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