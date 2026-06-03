"""DSL-DYAB-FULL — both RPN expansion + RCNN DyabDSL.

  RPN:  RPNExpandAssigner (static expansion, improved small-object recall)
  RCNN: DynamicSoftLabelAssignerDScaleDYAB (DyabDSL + static expansion
        + area-refine soft label)

Tests the combined effect of RPN and RCNN enhancements.
"""

# Merge both RPN and RCNN configs
_base_ = ['./dsl_dyab_rpn_default.py', './dsl_dyab_rcnn_default.py']
