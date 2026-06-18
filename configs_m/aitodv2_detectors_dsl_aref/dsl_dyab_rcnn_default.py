"""DSL-DYAB-RCNN — default (static expansion + DyabDSL) on Detectors.

RCNN-stage DynamicSoftLabelAssignerDScaleDYAB across all 3 cascade stages:
  expansion      – static
  dyab           – DyabDSL
  area-refine    – add_1, r_ref=32
"""

_base_ = ['./dsl_aref_default.py']

model = dict(
    train_cfg=dict(
        rcnn=[
            dict(
                _delete_=True,
                assigner=dict(
                    type='DynamicSoftLabelAssignerDScaleDYAB',
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
                    type='DynamicSoftLabelAssignerDScaleDYAB',
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
                    type='DynamicSoftLabelAssignerDScaleDYAB',
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
