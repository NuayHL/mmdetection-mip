"""DSL-DYAB-FULL — size-dependent expansion in both RPN and RCNN.

Same as dsl_dyab_full_default but with expansion_type='size_dependent'
in both stages.
"""

_base_ = ['./dsl_dyab_rpn_adapt.py', './dsl_dyab_rcnn_adapt.py']
