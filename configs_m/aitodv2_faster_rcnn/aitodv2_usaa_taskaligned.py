"""USAA variant — TaskAlignedAssigner instead of DynamicSoftLabelAssigner.

TaskAligned computes prior centers internally from ``(x1+x2)/2`` and
expects ``pred_instances.scores`` as probabilities. So:

  * ``prior_format='xyxy'`` (RPN proposals as-is, no point conversion)
  * ``cls_score_activation='sigmoid'`` so ``scores`` is in [0, 1]
"""

_base_ = ['./aitodv2_usaa.py']

model = dict(
    roi_head=dict(
        prior_format='xyxy',
        cls_score_activation='sigmoid',
    ),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                _delete_=True,
                type='TaskAlignedAssigner',
                topk=13),
        ),
    ),
)
