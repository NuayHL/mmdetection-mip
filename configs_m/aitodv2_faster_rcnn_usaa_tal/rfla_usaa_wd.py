"""RFLA-WD RPN + USAA Task-Aligned RCNN.

Same as ``rfla_usaa_kld.py`` but the RFLA RPN assigner uses Wasserstein
distance (``assign_metric='wd'``) instead of KL divergence.

Currency option: set the assigner ``align_mode``/``score_mode='wd'`` to match
the RPN currency. Default is ``iou`` (prediction-quality driven).
"""

_base_ = ['./rfla_usaa_kld.py']

model = dict(
    train_cfg=dict(
        rpn=dict(
            assigner=dict(assign_metric='wd'),
        ),
    ),
)