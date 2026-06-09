"""NWD-RKA + DSL-DYAB.

RPN:   NWD-RKA (RankingAssigner with NWD metric, topk=2)
RCNN:  DSL-DYAB (DynamicSoftLabelAssignerDScaleDYAB with DyabDSL)

This is the primary combined experiment.
"""

_base_ = ['./_base_nwdrka_dsl.py']
