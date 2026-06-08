"""RFLA-WD + DSL-DYAB.

Same as ``rfla_dsl_dyab.py`` but uses Wasserstein distance (WD) instead
of KL divergence as the RPN affinity metric.
"""

_base_ = ['./rfla_dsl_dyab.py']

model = dict(
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                assign_metric='wd',
            ),
        ),
    ),
)
