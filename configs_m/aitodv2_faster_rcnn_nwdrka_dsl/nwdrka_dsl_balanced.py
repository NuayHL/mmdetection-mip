"""NWD-RKA + DSL-DYAB with BALANCED (RandomSampler) RCNN sampling.

Same as ``nwdrka_dsl_dyab.py`` (RPN = NWD-RKA, RCNN = DynAssignRoIHead +
DynamicSoftLabelAssignerDScaleDYAB + QFL soft target) except the RCNN
sampler is swapped from ``PseudoSampler`` (keep ALL proposals → thousands of
easy negatives dominate the cls loss) to ``RandomSampler`` (the standard
StandardRoIHead 1:3 balanced subsampling, 512 RoIs).

Motivation: DSL was designed for dense one-stage heads where PseudoSampler +
focal loss handle imbalance. On a two-stage RCNN stacked on top of an
NWD-RKA RPN, the unbalanced PseudoSampler set was a prime suspect for the AP
drop. DynAssignRoIHead's sampler is NOT hardwired — it honours whatever
``train_cfg.rcnn.sampler`` specifies — so we keep its prediction-aware
two-pass adaptation (decoded bboxes + scores + point-format priors that DSL
needs) while restoring the balanced sampler.
"""

_base_ = ['./nwdrka_dsl_dyab.py']

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
