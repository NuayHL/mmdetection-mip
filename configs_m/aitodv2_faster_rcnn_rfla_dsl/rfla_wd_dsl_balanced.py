"""RFLA-WD + DSL-DYAB with BALANCED (RandomSampler) RCNN sampling.

Same as ``rfla_wd_dsl_dyab.py`` (RPN = RFLA Wasserstein distance, RCNN =
DynAssignRoIHead + DynamicSoftLabelAssignerDScaleDYAB + QFL soft target)
except the RCNN sampler is swapped from ``PseudoSampler`` to
``RandomSampler`` (standard 1:3 balanced subsampling, 512 RoIs).
"""

_base_ = ['./rfla_wd_dsl_dyab.py']

model = dict(
    train_cfg=dict(
        rcnn=dict(
            sampler=dict(
                _delete_=True,
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
        ),
    ),
)
