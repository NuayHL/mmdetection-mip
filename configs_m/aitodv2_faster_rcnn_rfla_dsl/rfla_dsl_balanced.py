"""RFLA-KLD + DSL-DYAB with BALANCED (RandomSampler) RCNN sampling.

Same as ``rfla_dsl_dyab.py`` (RPN = RFLA KLD, RCNN = DynAssignRoIHead +
DynamicSoftLabelAssignerDScaleDYAB + QFL soft target) except the RCNN
sampler is swapped from ``PseudoSampler`` to ``RandomSampler`` (standard
StandardRoIHead 1:3 balanced subsampling, 512 RoIs).

See ``../aitodv2_faster_rcnn_nwdrka_dsl/nwdrka_dsl_balanced.py`` for the
rationale: keep DSL's prediction-aware adaptation, restore balanced sampling.
"""

_base_ = ['./rfla_dsl_dyab.py']

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
