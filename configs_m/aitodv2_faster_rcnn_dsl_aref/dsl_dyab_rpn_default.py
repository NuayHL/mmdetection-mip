"""DSL-DYAB-RPN — default (static expansion in RPN).

Replaces RPN MaxIoUAssigner with RPNExpandAssigner:
  - Anchors whose centres lie up to scale_ratio×stride outside a GT
    box are still candidates (expansion for small-object recall).
  - Standard MaxIoU thresholds applied on the expanded candidate pool.
  - Dyab + area-refine are NOT used in RPN (no cls/bbox predictions yet).

Keeps RCNN stage unchanged (area-refine only, via dsl_aref_default).
"""

_base_ = ['./dsl_aref_default.py']

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
