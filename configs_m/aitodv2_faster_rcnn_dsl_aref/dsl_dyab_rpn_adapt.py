"""DSL-DYAB-RPN — size-dependent expansion in RPN.

expansion_type='size_dependent': smaller objects get more expansion.
Tests whether making RPN expansion adaptive further helps.
"""

_base_ = ['./dsl_dyab_rpn_default.py']

model = dict(
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                expansion_type='size_dependent',
            ),
        ),
    ),
)
