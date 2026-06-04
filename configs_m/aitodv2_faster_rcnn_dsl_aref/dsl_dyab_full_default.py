"""DSL-DYAB-FULL — both RPN expansion + RCNN DyabDSL.

  RPN:  RPNExpandAssigner (static expansion, improved small-object recall)
  RCNN: DynamicSoftLabelAssignerDScaleDYAB (DyabDSL + static expansion
        + area-refine soft label)

Tests the combined effect of RPN and RCNN enhancements.

NOTE: we inherit from dsl_dyab_rcnn_default (RCNN changes) and manually
merge the RPN train_cfg from dsl_dyab_rpn_default.  Diamond inheritance
via _base_ would pull in _base_.py twice, causing a duplicate-key error.
"""

_base_ = ['./dsl_dyab_rcnn_default.py']

# Manually merge RPN overrides from dsl_dyab_rpn_default
model = dict(
    train_cfg=dict(
        rpn=dict(
            _delete_=True,
            assigner=dict(
                type='RPNExpandAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
                scale_ratio=1.0,
                base_scale=8,
                expansion_type='static',
                expansion_r_ref=32.0,
            ),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
    ),
)
