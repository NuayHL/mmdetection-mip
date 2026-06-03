#!/usr/bin/env python
"""Standalone evaluation script for AITODv2 Faster R-CNN models.

Uses ``AITODCocoMetric`` (custom area ranges: verytiny/tiny/small/medium
and maxDets=[1, 100, 1500]) and produces calibration-analysis plots:

1. **Score-Precision calibration** at IoU=0.5:
   how well confidence scores align with actual precision.

2. **Prediction score vs prediction area** scatter plot:
   scores vs bbox areas, coloured by TP/FP status at IoU=0.5.

Intermediate prediction data (with TP/FP labels) is saved as a pickle file
so it can be reloaded later for additional plotting or cross-experiment
comparison without re-running inference.

Usage::

    python eval_aitod.py --checkpoint /path/to/epoch_24.pth
    python eval_aitod.py --checkpoint /path/to/epoch_24.pth --out-dir my_eval
"""

import argparse
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

# Ensure the project root is on sys.path so that ``mmdet`` is importable.
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mmengine.config import Config
from mmengine.runner import Runner

from mmdet.datasets.api_wrappers import COCO
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo

# Lazy matplotlib imports (only when plotting is needed).
_plt = None
_mlp = None  # matplotlib colormaps


def _ensure_mpl():
    """Late-import matplotlib so the script stays usable headless."""
    global _plt, _mlp
    if _plt is None:
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend
        import matplotlib.pyplot as _plt
        import matplotlib as _mlp


# ---------------------------------------------------------------------------
#  Bbox utilities
# ---------------------------------------------------------------------------

def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert [x, y, w, h] → [x1, y1, x2, y2]."""
    out = np.array(boxes, dtype=np.float32, copy=True)
    out[..., 2] = out[..., 0] + out[..., 2]
    out[..., 3] = out[..., 1] + out[..., 3]
    return out


def _box_area_xywh(boxes: np.ndarray) -> np.ndarray:
    """Area of [x, y, w, h] boxes."""
    return boxes[..., 2] * boxes[..., 3]


def _box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of **xywh** boxes.

    Returns:
        ``(len(boxes_a), len(boxes_b))`` float array.
    """
    # Compute areas from the *original* xywh representation.
    area_a = boxes_a[..., 2] * boxes_a[..., 3]
    area_b = boxes_b[..., 2] * boxes_b[..., 3]

    a = _xywh_to_xyxy(boxes_a)
    b = _xywh_to_xyxy(boxes_b)

    # Intersection
    lt = np.maximum(a[:, np.newaxis, :2], b[np.newaxis, :, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[np.newaxis, :, 2:])
    wh = np.maximum(0.0, rb - lt)
    inter = wh[:, :, 0] * wh[:, :, 1]

    # Union
    union = area_a[:, np.newaxis] + area_b[np.newaxis, :] - inter
    union = np.maximum(union, 1e-12)
    return inter / union


# ---------------------------------------------------------------------------
#  Matching logic (TP / FP per prediction at a fixed IoU)
# ---------------------------------------------------------------------------

def _greedy_match(coco_gt: COCO, predictions: list,
                  iou_thr: float = 0.5) -> list:
    """Assign TP / FP label to every prediction (greedy, per img × cat).

    Matches are performed *independently* for each ``(image_id, category_id)``
    pair.  Inside a group predictions are sorted by score (descending); each is
    matched to the unmatched GT with the highest IoU ≥ *iou_thr*.

    Parameters
    ----------
    coco_gt : COCO
        Ground-truth COCO API instance.
    predictions : list[dict]
        COCO-format predictions.  Each dict must have keys
        ``image_id``, ``category_id``, ``score``, ``bbox`` (xywh).
    iou_thr : float
        IoU threshold for a true positive.

    Returns
    -------
    list[dict]
        Each element is a copy of a prediction dict with these keys added:
        ``is_tp`` (bool), ``max_iou`` (float),
        ``gt_area`` (float | None — area of the matched GT box, or None).
    """
    # Index GTs by (img_id, cat_id)
    gt_by_key: dict[tuple, list] = defaultdict(list)
    for ann in coco_gt.loadAnns(coco_gt.getAnnIds()):
        if ann.get('iscrowd', 0):
            continue
        gt_by_key[(ann['image_id'], ann['category_id'])].append(ann)

    # Group predictions
    pred_by_key: dict[tuple, list] = defaultdict(list)
    for i, p in enumerate(predictions):
        key = (p['image_id'], p['category_id'])
        pred_by_key[key].append((i, p))  # keep original index for ordering later

    results = [None] * len(predictions)

    for (img_id, cat_id), items in pred_by_key.items():
        gts = gt_by_key.get((img_id, cat_id), [])

        # Sort by score descending
        items_sorted = sorted(items, key=lambda x: x[1]['score'], reverse=True)

        if len(gts) == 0:
            # No GT of this category in this image → every pred is FP
            for orig_idx, p in items_sorted:
                results[orig_idx] = {**p, 'is_tp': False, 'max_iou': 0.0,
                                     'gt_area': None}
            continue

        gt_boxes = np.array([g['bbox'] for g in gts], dtype=np.float32)
        gt_matched = np.zeros(len(gts), dtype=bool)

        for orig_idx, p in items_sorted:
            pred_box = np.array(p['bbox'], dtype=np.float32).reshape(1, 4)
            ious = _box_iou_matrix(pred_box, gt_boxes)[0]  # shape (n_gt,)
            ious[gt_matched] = -1.0
            best_idx = int(np.argmax(ious))
            best_iou = float(ious[best_idx])

            is_tp = best_iou >= iou_thr
            gt_area = float(_box_area_xywh(
                gt_boxes[best_idx:best_idx + 1])[0]) if is_tp else None

            if is_tp:
                gt_matched[best_idx] = True

            results[orig_idx] = {**p, 'is_tp': is_tp, 'max_iou': best_iou,
                                 'gt_area': gt_area}

    return results


# ---------------------------------------------------------------------------
#  Calibration metrics & plotting
# ---------------------------------------------------------------------------

def _compute_calibration_curve(matched_preds: list, n_bins: int = 20):
    """Compute binned calibration curve (score vs precision at IoU=0.5).

    Returns
    -------
    bin_centers : np.ndarray  (n_bins,)
    bin_precisions : np.ndarray  (n_bins,) — precision in each bin
    bin_counts : np.ndarray  (n_bins,) — number of predictions in each bin
    ece : float  — Expected Calibration Error
    """
    scores = np.array([r['score'] for r in matched_preds], dtype=np.float64)
    is_tp = np.array([r['is_tp'] for r in matched_preds], dtype=np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_precisions = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = (scores >= bin_edges[i]) & (scores < bin_edges[i + 1])
        bin_counts[i] = mask.sum()
        if bin_counts[i] > 0:
            bin_precisions[i] = is_tp[mask].mean()

    # ECE: weighted average of |precision - confidence|
    total = len(scores)
    ece = np.sum(bin_counts / total * np.abs(bin_precisions - bin_centers))

    return bin_centers, bin_precisions, bin_counts, float(ece)


def _plot_calibration(matched_preds: list, out_path: str, model_label: str = ''):
    """Save score-precision calibration plot (IoU=0.5)."""
    _ensure_mpl()

    bin_centers, bin_precs, bin_counts, ece = _compute_calibration_curve(
        matched_preds, n_bins=20)

    fig, (ax1, ax2) = _plt.subplots(1, 2, figsize=(14, 5.5))

    # ---- Left: binned calibration curve ----
    ideal_bins = np.linspace(0, 1, 21)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Perfect calibration')
    ax1.plot(bin_centers, bin_precs, 'o-', color='#1f77b4', markersize=7,
             label=f'{model_label}  ECE={ece:.4f}')
    ax1.fill_between(bin_centers, bin_precs, bin_centers, alpha=0.15,
                     color='#1f77b4')
    ax1.set_xlabel('Confidence score', fontsize=12)
    ax1.set_ylabel('Precision @IoU=0.5', fontsize=12)
    ax1.set_title('Reliability diagram (score vs precision)', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    # Annotate bin counts
    for x, y, c in zip(bin_centers, bin_precs, bin_counts):
        if c > 0:
            ax1.text(x, y + 0.03, str(c), ha='center', fontsize=7,
                     color='gray', alpha=0.8)

    # ---- Right: cumulative precision curve ----
    scores = np.array([r['score'] for r in matched_preds], dtype=np.float64)
    is_tp = np.array([r['is_tp'] for r in matched_preds], dtype=np.float64)
    order = np.argsort(-scores)  # descending
    scores_desc = scores[order]
    tp_cum = np.cumsum(is_tp[order])

    # Evaluate at 200 evenly-spaced thresholds
    thrs = np.linspace(0.0, 1.0, 200)
    cum_prec = np.zeros_like(thrs)
    cum_recall = np.zeros_like(thrs)
    total_tp = is_tp.sum()
    for i, thr in enumerate(thrs):
        mask = scores_desc >= thr
        n = mask.sum()
        if n > 0:
            cum_prec[i] = tp_cum[mask].sum() / n
            cum_recall[i] = tp_cum[mask].sum() / max(total_tp, 1)

    ax2.plot(thrs, cum_prec, 'b-', linewidth=2, label='Precision')
    ax2.plot(thrs, cum_recall, 'r-', linewidth=2, label='Recall')
    ax2.set_xlabel('Score threshold τ  (preds with score ≥ τ)', fontsize=12)
    ax2.set_ylabel('Precision / Recall @IoU=0.5', fontsize=12)
    ax2.set_title('Cumulative precision & recall vs score threshold', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'Calibration Analysis — IoU=0.5  {model_label}',
                 fontsize=14, fontweight='bold')
    _plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    _plt.close(fig)
    print(f'  Calibration plot saved → {out_path}')


def _plot_score_area(matched_preds: list, out_path: str, model_label: str = ''):
    """Save prediction score vs prediction area scatter plot."""
    _ensure_mpl()

    scores = np.array([r['score'] for r in matched_preds], dtype=np.float64)
    areas = np.array([r['bbox'][2] * r['bbox'][3] for r in matched_preds],
                     dtype=np.float64)
    is_tp = np.array([r['is_tp'] for r in matched_preds], dtype=bool)

    tp_mask = is_tp
    fp_mask = ~is_tp

    fig, (ax1, ax2) = _plt.subplots(1, 2, figsize=(14, 5.5))

    # ---- Left: scatter, log-log scale ----
    ax1.scatter(areas[tp_mask], scores[tp_mask], s=3, alpha=0.35,
                c='#2ca02c', label=f'TP  (n={tp_mask.sum():,})', edgecolors='none')
    ax1.scatter(areas[fp_mask], scores[fp_mask], s=3, alpha=0.25,
                c='#d62728', label=f'FP  (n={fp_mask.sum():,})', edgecolors='none')
    ax1.set_xscale('log')
    ax1.set_xlabel('Prediction area (px²) — log scale', fontsize=12)
    ax1.set_ylabel('Confidence score', fontsize=12)
    ax1.set_title('Score vs area (log-log)', fontsize=13)
    ax1.legend(fontsize=9, markerscale=3)
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=0.3)

    # ---- Right: binned area vs avg score ----
    if len(areas) > 0:
        min_log = np.log10(max(areas.min(), 1e-2))  # prevent log(0)
        max_log = np.log10(areas.max())
        bins = 10 ** np.linspace(min_log, max_log, 21)
    else:
        bins = np.linspace(0, 1, 21)

    bin_centers_area = np.sqrt(bins[:-1] * bins[1:])  # geometric mean
    avg_scores = np.zeros(len(bin_centers_area))
    tp_rates = np.zeros(len(bin_centers_area))
    bin_counts_area = np.zeros(len(bin_centers_area), dtype=int)

    for i in range(len(bins) - 1):
        mask = (areas >= bins[i]) & (areas < bins[i + 1])
        bin_counts_area[i] = mask.sum()
        if bin_counts_area[i] > 0:
            avg_scores[i] = scores[mask].mean()
            tp_rates[i] = is_tp[mask].mean()

    ax2.plot(bin_centers_area, avg_scores, 'b-o', markersize=5, linewidth=1.5,
             label='Avg score')
    ax2.plot(bin_centers_area, tp_rates, 'g-s', markersize=5, linewidth=1.5,
             label='Precision @IoU=0.5')
    ax2.set_xscale('log')
    ax2.set_xlabel('Prediction area (px²) — log scale', fontsize=12)
    ax2.set_ylabel('Avg score / Precision', fontsize=12)
    ax2.set_title('Avg score & precision by area bin', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(True, alpha=0.3)

    # Annotate bin counts
    for x, y, c in zip(bin_centers_area, avg_scores, bin_counts_area):
        if c > 0:
            ax2.text(x, y + 0.04, str(c), ha='center', fontsize=6.5,
                     color='gray', alpha=0.8)

    fig.suptitle(f'Score–Area Analysis — IoU=0.5  {model_label}',
                 fontsize=14, fontweight='bold')
    _plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    _plt.close(fig)
    print(f'  Score-area plot saved   → {out_path}')


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a Faster R-CNN model with AITODCocoMetric '
                    '+ calibration analysis')

    parser.add_argument(
        '--config',
        type=str,
        default='configs_m/aitodv2_faster_rcnn/aitodv2_iou.py',
        help='Path to the training config (default: aitodv2_iou baseline)')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to the .pth checkpoint to evaluate')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='eval_outputs',
        help='Directory to save all outputs (pred json, pkl, plots)')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Job launcher (default: none)')
    parser.add_argument(
        '--local_rank', '--local-rank',
        type=int,
        default=0,
        help='Local rank for distributed testing')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Setup
    # ------------------------------------------------------------------
    setup_cache_size_limit_of_dynamo()
    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. Load config and set AITOD evaluator + checkpoint
    # ------------------------------------------------------------------
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher

    pred_json_prefix = os.path.join(args.out_dir, 'aitodv2_preds')
    ann_file = 'data/ai-todv2/annotations_v2/aitodv2_val_fixed.json'

    aitod_evaluator = dict(
        type='AITODCocoMetric',
        ann_file=ann_file,
        metric='bbox',
        proposal_nums=(1, 100, 1500),
        format_only=False,
        backend_args=None,
        classwise=True,
        outfile_prefix=pred_json_prefix,
    )
    cfg.test_evaluator = aitod_evaluator
    cfg.val_evaluator = aitod_evaluator
    cfg.load_from = args.checkpoint
    cfg.work_dir = args.out_dir

    # ------------------------------------------------------------------
    # 3. Run standard AITOD evaluation via the Runner
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Config:     {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {args.out_dir}")
    print(f"Evaluator:  AITODCocoMetric (verytiny/tiny/small/medium)")
    print(f"{'='*60}\n")

    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    runner.test()

    # ------------------------------------------------------------------
    # 4. Load COCO-format predictions and ground truth for analysis
    # ------------------------------------------------------------------
    pred_json_path = f'{pred_json_prefix}.bbox.json'
    if not os.path.exists(pred_json_path):
        print(f"\n[WARN] Prediction JSON not found at {pred_json_path}. "
              "Skipping calibration analysis.")
        return

    import json
    with open(pred_json_path) as f:
        raw_predictions = json.load(f)

    if len(raw_predictions) == 0:
        print("\n[WARN] Zero predictions — skipping calibration analysis.")
        return

    coco_gt = COCO(ann_file)
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_gt.loadCats(coco_gt.getCatIds())}

    # ------------------------------------------------------------------
    # 5. Match predictions to GT (greedy, per image×category, IoU=0.5)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Calibration analysis: matching predictions to GT  (IoU=0.5)")
    print(f"{'='*60}")

    print(f"  Predictions : {len(raw_predictions):,}")
    print(f"  GT boxes    : {len(coco_gt.loadAnns(coco_gt.getAnnIds())):,}")
    matched = _greedy_match(coco_gt, raw_predictions, iou_thr=0.5)

    n_tp = sum(1 for r in matched if r['is_tp'])
    n_fp = len(matched) - n_tp
    print(f"  TP          : {n_tp:,}")
    print(f"  FP          : {n_fp:,}")
    print(f"  Precision   : {n_tp / len(matched):.4f}  (at IoU=0.5, all scores)")

    # ------------------------------------------------------------------
    # 6. Save intermediate data for later reuse
    # ------------------------------------------------------------------
    pkl_path = os.path.join(args.out_dir, 'calibration_data.pkl')
    # Add lightweight metadata for later cross-experiment comparison
    pkl_data = {
        'model_name': os.path.splitext(os.path.basename(args.checkpoint))[0],
        'config': args.config,
        'checkpoint': args.checkpoint,
        'matched_predictions': matched,
        'cat_id_to_name': cat_id_to_name,
        'iou_thr': 0.5,
    }
    with open(pkl_path, 'wb') as f:
        pickle.dump(pkl_data, f)
    print(f"\n  Intermediate data saved → {pkl_path}")
    print(f"    (load with: pickle.load(open('{pkl_path}', 'rb')))")

    # ------------------------------------------------------------------
    # 7. Generate plots
    # ------------------------------------------------------------------
    model_label = os.path.splitext(os.path.basename(args.checkpoint))[0]

    cal_path = os.path.join(args.out_dir, 'calibration_score_precision.png')
    _plot_calibration(matched, cal_path, model_label=model_label)

    area_path = os.path.join(args.out_dir, 'score_vs_area.png')
    _plot_score_area(matched, area_path, model_label=model_label)

    print(f"\n{'='*60}")
    print(f"All done. Outputs in: {args.out_dir}/")
    print(f"  - {os.path.basename(pred_json_path)}   (COCO-format predictions)")
    print(f"  - calibration_data.pkl                 (matched preds, reusable)")
    print(f"  - calibration_score_precision.png      (score-precision plot)")
    print(f"  - score_vs_area.png                    (score-area scatter)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
