"""RFLA-KLD + DSL-DYAB.

RPN:   RFLA (RFGenerator + HieAssigner-KLD, topk=[3,1], ratio=0.9)
RCNN:  DSL-DYAB (DynamicSoftLabelAssignerDScaleDYAB with DyabDSL)

This is the primary combined experiment.
"""

_base_ = ['./_base_rfla_dsl.py']
