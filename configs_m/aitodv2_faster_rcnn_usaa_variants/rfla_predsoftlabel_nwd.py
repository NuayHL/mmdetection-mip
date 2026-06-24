"""RFLA-KLD RPN + prediction-aware soft-label RCNN, NWD currency  (Branch B).

Strategy B — fix APvt / APt by changing the soft-label *currency* from IoU to
NWD (tiny-object friendly), instead of crudely lifting IoU with add_1.

Diagnosis of ``rfla_predsoftlabel.py`` (score_mode='iou', add_1, floor=0.5):
AP50 ↑ but mAP / APt / APvt ↓ and still over-confident. Two inflation sources
hurt the high-IoU / tiny regime:
  * floor=pos_iou_thr(0.5) clamps poorly-localised positives UP (AP50↑, mAP↓),
  * add_1 over-lifts mid-IoU yet still leaves tiny targets *below* normal ones.

Fix here (single coherent change of strategy):
  * score_mode='nwd' — for a tiny object the decoded prediction often has low,
    noisy IoU but a well-aligned centre; NWD ∈ [0,1] gives it a *fair* quality
    target (and matches the KLD RPN's currency). This lifts tiny scores in a
    principled, size-aware way → better APvt / APt ranking.
  * calibrate_mode='none' — NWD is already size-fair, so the add_1 lift would
    double-inflate and re-create the over-confidence. Drop it.
  * floor_value=0.1 — a *gentle* floor (not 0.5) for early-training stability
    only; honest low scores are preserved (add_gt_as_proposals=True already
    guarantees a target=1.0 positive per GT, so this is safe).

Compare against vanilla RFLA-KLD and against ``rfla_predsoftlabel.py``.
"""

_base_ = ['./rfla_predsoftlabel.py']

model = dict(
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                score_mode='nwd',
                calibrate_mode='none',
                floor_to_pos_thr=True,
                floor_value=0.1),
        ),
    ),
)
