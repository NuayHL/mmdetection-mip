"""NWD-RKA (Normalized Wasserstein Distance + Ranking-based K Assignment)
base config for AITOD-v2 Faster R-CNN experiments.

Reference: "Detecting Tiny Objects in Aerial Images via Normalized
Wasserstein Distance and a New Benchmark" (AI-TOD-v2).

Key difference from the MaxIoU baseline:
  * RPN assigner: RankingAssigner (RKA) — for each GT, assigns the top-k
    proposals ranked by the NWD similarity metric as positives, instead of
    IoU thresholding. NWD is scale-invariant, which is critical for tiny
    objects where IoU is unstable.

The anchor generator stays the standard AnchorGenerator (NWD-RKA does NOT
change anchors — that's RFLA's RFGenerator). All other settings (dataset,
schedule, RPN proposals, RoI head, test) are inherited from ``_base_.py``.
"""

_base_ = ['../aitodv2_faster_rcnn/_base_.py']

# `constant` is the NWD normalization constant C. The paper uses the
# dataset-average absolute size; 12.7 is the AI-TOD value.
model = dict(
    train_cfg=dict(
        rpn=dict(
            # Replace MaxIoUAssigner with RankingAssigner (RKA) using NWD.
            assigner=dict(
                _delete_=True,
                type='RankingAssigner',
                ignore_iof_thr=-1,
                gpu_assign_thr=512,
                iou_calculator=dict(type='BboxDistanceMetric', constant=12.7),
                assign_metric='nwd',
                topk=2),
        ),
    ),
)
