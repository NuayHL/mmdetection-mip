#!/usr/bin/env python
"""Standalone evaluation script for AITODv2 Faster R-CNN models.

Uses ``AITODCocoMetric`` (custom area ranges: verytiny/tiny/small/medium
and maxDets=[1, 100, 1500]) and produces calibration-analysis plots:

1. **Score-Precision calibration** at IoU=0.5 (with per-area-range ECE).
2. **Prediction score vs prediction area** scatter plot (area computed in
   *resized / model-input* pixel space so object sizes reflect what the
   model actually sees).

Intermediate prediction data (with TP/FP labels, original AND resized
areas) is saved as a pickle file for later cross-experiment comparison
without re-running inference.

Usage — single model eval::

    python eval_aitod.py --checkpoint /path/to/epoch_24.pth
    python eval_aitod.py --checkpoint /path/to/epoch_24.pth --out-dir my_eval

Usage — compare two experiments::

    python eval_aitod.py --load eval_outputs/exp1/calibration_data.pkl \\
                                  eval_outputs/exp2/calibration_data.pkl \\
                         --labels "Baseline" "Our Method" \\
                         --area-key resized --n-bins 10
"""

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

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

# ---------------------------------------------------------------------------
#  Constants — AITOD area ranges (squared pixels in *original* image space)
# ---------------------------------------------------------------------------
AITOD_AREA_RANGES = [
    ('all',       0 ** 2,  1e5 ** 2),
    ('verytiny',  0 ** 2,  8 ** 2),    # [0,   64)
    ('tiny',      8 ** 2,  16 ** 2),   # [64,  256)
    ('small',     16 ** 2, 32 ** 2),   # [256, 1024)
    ('medium',    32 ** 2, 1e5 ** 2),  # [1024, 1e10)
]

# Lazy matplotlib imports (only when plotting is needed).
_plt = None


def _ensure_mpl():
    """Late-import matplotlib so the script stays usable headless."""
    global _plt
    if _plt is None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as _plt


# Color palette for multi-model comparison (consistent with ultralytics style)
COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']

# Fixed axis limits for fair comparison across models
X_LIM_LOG = (1, 10 ** 4)      # resized px² area range (log scale)
Y_LIM = (-0.02, 1.02)


def _apply_pub_style():
    """Apply publication-quality rcParams consistent with ultralytics style."""
    _ensure_mpl()
    _plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 13,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'legend.framealpha': 0.6,
        'legend.edgecolor': '0.5',
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linestyle': '--',
        'lines.linewidth': 1.5,
    })


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
    area_a = boxes_a[..., 2] * boxes_a[..., 3]
    area_b = boxes_b[..., 2] * boxes_b[..., 3]

    a = _xywh_to_xyxy(boxes_a)
    b = _xywh_to_xyxy(boxes_b)

    lt = np.maximum(a[:, np.newaxis, :2], b[np.newaxis, :, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[np.newaxis, :, 2:])
    wh = np.maximum(0.0, rb - lt)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area_a[:, np.newaxis] + area_b[np.newaxis, :] - inter
    union = np.maximum(union, 1e-12)
    return inter / union


# ---------------------------------------------------------------------------
#  Resize scale helpers
# ---------------------------------------------------------------------------

def _extract_resize_max_size(cfg) -> float:
    """Extract the target max size from the test-pipeline Resize transform.

    Defaults to 800 if the pipeline cannot be parsed.
    """
    try:
        pipe = cfg.test_dataloader.dataset.pipeline
        for step in pipe:
            if step.get('type') == 'Resize':
                scale = step['scale']
                if isinstance(scale, (list, tuple)):
                    return float(max(scale))
                return float(scale)
    except Exception:
        pass
    return 800.0


def _build_scale_map(coco_gt: COCO, max_size: float) -> dict:
    """Build ``img_id → resize_scale`` map.

    For ``Resize(scale=(S, S), keep_ratio=True)`` the scale factor is::

        scale = S / max(img_height, img_width)

    Returns
    -------
    dict  img_id → float (scale factor, ≤ 1.0)
    """
    scale_map = {}
    for img_id, img_info in coco_gt.imgs.items():
        h, w = img_info['height'], img_info['width']
        scale_map[img_id] = max_size / max(h, w)
    return scale_map


def _classify_area(area: float) -> str:
    """Return the AITOD area-range label for a raw *original-space* area.

    ``all`` is intentionally skipped here — every box belongs to ``all``
    by definition; we want the *most specific* label.
    """
    for label, lo, hi in AITOD_AREA_RANGES:
        if label == 'all':
            continue
        if lo <= area < hi:
            return label
    return 'unknown'


# ---------------------------------------------------------------------------
#  Matching logic (TP / FP per prediction at a fixed IoU)
# ---------------------------------------------------------------------------

def _greedy_match(coco_gt: COCO, predictions: list,
                  scale_map: dict,
                  iou_thr: float = 0.5) -> list:
    """Assign TP / FP label to every prediction (greedy, per img × cat).

    Additionally computes both *original* and *resized* (model-input) areas,
    and assigns each prediction an AITOD area-range label based on its
    resized bbox area.

    Parameters
    ----------
    coco_gt : COCO
    predictions : list[dict]
        COCO-format dicts: ``image_id``, ``category_id``, ``score``,
        ``bbox`` (xywh).
    scale_map : dict
        ``img_id → resize_scale`` (from ``_build_scale_map``).
    iou_thr : float

    Returns
    -------
    list[dict]
        Predictions augmented with:
        ``is_tp``, ``max_iou``,
        ``area_original`` — pred bbox area in original image px²,
        ``area_resized`` — pred bbox area in resized / model-input px²,
        ``gt_area_original`` — matched GT area in original px² (or None),
        ``gt_area_resized`` — matched GT area in resized px² (or None),
        ``area_label`` — AITOD label for ``area_resized``.
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
        pred_by_key[key].append((i, p))

    results = [None] * len(predictions)

    for (img_id, cat_id), items in pred_by_key.items():
        gts = gt_by_key.get((img_id, cat_id), [])
        scale = scale_map.get(img_id, 1.0)
        scale2 = scale * scale

        items_sorted = sorted(items, key=lambda x: x[1]['score'], reverse=True)

        if len(gts) == 0:
            for orig_idx, p in items_sorted:
                p_area_orig = p['bbox'][2] * p['bbox'][3]
                results[orig_idx] = {
                    **p,
                    'is_tp': False, 'max_iou': 0.0,
                    'area_original': float(p_area_orig),
                    'area_resized': float(p_area_orig * scale2),
                    'gt_area_original': None,
                    'gt_area_resized': None,
                    'area_label': _classify_area(p_area_orig * scale2),
                }
            continue

        gt_boxes = np.array([g['bbox'] for g in gts], dtype=np.float32)
        gt_areas_orig = _box_area_xywh(gt_boxes)
        gt_matched = np.zeros(len(gts), dtype=bool)

        for orig_idx, p in items_sorted:
            pred_box = np.array(p['bbox'], dtype=np.float32).reshape(1, 4)
            ious = _box_iou_matrix(pred_box, gt_boxes)[0]
            ious[gt_matched] = -1.0
            best_idx = int(np.argmax(ious))
            best_iou = float(ious[best_idx])

            is_tp = best_iou >= iou_thr
            p_area_orig = float(p['bbox'][2] * p['bbox'][3])

            if is_tp:
                gt_matched[best_idx] = True
                gt_a_orig = float(gt_areas_orig[best_idx])
            else:
                gt_a_orig = None

            results[orig_idx] = {
                **p,
                'is_tp': is_tp,
                'max_iou': best_iou,
                'area_original': p_area_orig,
                'area_resized': float(p_area_orig * scale2),
                'gt_area_original': gt_a_orig,
                'gt_area_resized': float(gt_a_orig * scale2) if gt_a_orig is not None else None,
                'area_label': _classify_area(p_area_orig * scale2),
            }

    return results


# ---------------------------------------------------------------------------
#  Calibration metrics & plotting
# ---------------------------------------------------------------------------

def _compute_calibration_curve(matched_preds: list, n_bins: int = 20):
    """Compute binned calibration curve (score vs precision at IoU=0.5).

    Returns
    -------
    bin_centers, bin_precisions, bin_counts : np.ndarray
    ece : float
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

    total = len(scores)
    ece = np.sum(bin_counts / total * np.abs(bin_precisions - bin_centers))

    return bin_centers, bin_precisions, bin_counts, float(ece)


def _per_area_calibration(matched_preds: list) -> dict:
    """Compute ECE and precision per AITOD area range (based on resized area).

    Returns
    -------
    dict  area_label → {'ece': float, 'precision': float, 'n_preds': int,
                         'n_tp': int, 'n_gt': int}
    """
    # Group by area_label
    by_label = defaultdict(list)
    for r in matched_preds:
        by_label[r['area_label']].append(r)

    result = {}
    for label in ['verytiny', 'tiny', 'small', 'medium']:
        preds = by_label.get(label, [])
        if len(preds) == 0:
            result[label] = {'ece': float('nan'), 'precision': float('nan'),
                             'n_preds': 0, 'n_tp': 0}
            continue
        _, _, _, ece = _compute_calibration_curve(preds, n_bins=10)
        n_tp = sum(1 for r in preds if r['is_tp'])
        result[label] = {
            'ece': ece,
            'precision': n_tp / len(preds) if len(preds) > 0 else float('nan'),
            'n_preds': len(preds),
            'n_tp': n_tp,
        }
    return result


def _compute_calibration_full(matched_preds: list, n_bins: int = 10) -> dict:
    """Compute calibration dict with ECE and MCE (ultralytics-style).

    Uses **mean confidence per bin** (``avg_conf``) rather than bin centres
    for ECE/MCE — this is more faithful to the definition.

    Returns
    -------
    dict  with keys: bin_edges, bin_centers, precision, avg_conf, count,
          ece, mce.  ``None`` if no predictions.
    """
    preds = matched_preds  # all preds are either TP or FP
    if not preds:
        return None

    scores = np.array([r['score'] for r in preds], dtype=np.float64)
    is_tp = np.array([r['is_tp'] for r in preds], dtype=np.float64)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    precision = np.zeros(n_bins)
    avg_conf = np.zeros(n_bins)
    count = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = (scores >= bin_edges[i]) & (scores < bin_edges[i + 1])
        count[i] = mask.sum()
        if count[i] > 0:
            precision[i] = is_tp[mask].mean()
            avg_conf[i] = scores[mask].mean()
        else:
            precision[i] = 0.0
            avg_conf[i] = bin_centers[i]

    total = len(preds)
    ece = float(np.sum(count / total * np.abs(precision - avg_conf)))
    mce = float(np.max(np.abs(precision - avg_conf)))

    return {
        'bin_edges': bin_edges,
        'bin_centers': bin_centers,
        'precision': precision,
        'avg_conf': avg_conf,
        'count': count,
        'ece': ece,
        'mce': mce,
    }


def _per_area_calibration_full(matched_preds: list, n_bins: int = 10) -> dict:
    """Per-area-range calibration with **both** ECE and MCE.

    Returns
    -------
    dict  area_label → {'ece', 'mce', 'precision', 'n_preds', 'n_tp'}
    """
    by_label = defaultdict(list)
    for r in matched_preds:
        by_label[r['area_label']].append(r)

    result = {}
    for label in ['verytiny', 'tiny', 'small', 'medium']:
        preds = by_label.get(label, [])
        if len(preds) == 0:
            result[label] = {
                'ece': float('nan'), 'mce': float('nan'),
                'precision': float('nan'), 'n_preds': 0, 'n_tp': 0,
            }
            continue
        cal = _compute_calibration_full(preds, n_bins=n_bins)
        n_tp = sum(1 for r in preds if r['is_tp'])
        result[label] = {
            'ece': cal['ece'] if cal else float('nan'),
            'mce': cal['mce'] if cal else float('nan'),
            'precision': n_tp / len(preds) if len(preds) > 0 else float('nan'),
            'n_preds': len(preds),
            'n_tp': n_tp,
        }
    return result


# ---------------------------------------------------------------------------
#  Single-model calibration & score-area plotting
# ---------------------------------------------------------------------------


def _plot_calibration(matched_preds: list, out_path: str, model_label: str = ''):
    """Save score-precision calibration plot with per-area-range detail."""
    _ensure_mpl()

    # ---- Per-area-range ECE stats ----
    per_area = _per_area_calibration(matched_preds)

    # ---- Global calibration curve ----
    bin_centers, bin_precs, bin_counts, ece = _compute_calibration_curve(
        matched_preds, n_bins=20)

    fig = _plt.figure(figsize=(16, 10))

    # -- Top-left: global reliability diagram --
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Perfect calibration')
    ax1.plot(bin_centers, bin_precs, 'o-', color='#1f77b4', markersize=7,
             label=f'All   ECE={ece:.4f}')
    ax1.fill_between(bin_centers, bin_precs, bin_centers, alpha=0.15,
                     color='#1f77b4')
    ax1.set_xlabel('Confidence score', fontsize=11)
    ax1.set_ylabel('Precision @IoU=0.5', fontsize=11)
    ax1.set_title('Reliability diagram (all objects)', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    for x, y, c in zip(bin_centers, bin_precs, bin_counts):
        if c > 0:
            ax1.text(x, y + 0.03, str(c), ha='center', fontsize=6.5,
                     color='gray', alpha=0.8)

    # -- Top-right: cumulative precision/recall --
    ax2 = fig.add_subplot(2, 2, 2)
    scores = np.array([r['score'] for r in matched_preds], dtype=np.float64)
    is_tp = np.array([r['is_tp'] for r in matched_preds], dtype=np.float64)
    order = np.argsort(-scores)
    scores_desc = scores[order]
    tp_cum = np.cumsum(is_tp[order])
    total_tp = is_tp.sum()

    thrs = np.linspace(0.0, 1.0, 200)
    cum_prec = np.zeros_like(thrs)
    cum_recall = np.zeros_like(thrs)
    for i, thr in enumerate(thrs):
        mask = scores_desc >= thr
        n = mask.sum()
        if n > 0:
            cum_prec[i] = tp_cum[mask].sum() / n
            cum_recall[i] = tp_cum[mask].sum() / max(total_tp, 1)

    ax2.plot(thrs, cum_prec, 'b-', linewidth=2, label='Precision')
    ax2.plot(thrs, cum_recall, 'r-', linewidth=2, label='Recall')
    ax2.set_xlabel('Score threshold τ  (preds with score ≥ τ)', fontsize=11)
    ax2.set_ylabel('Precision / Recall @IoU=0.5', fontsize=11)
    ax2.set_title('Cumulative precision & recall vs score threshold', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    # -- Bottom-left: per-area-range reliability curves --
    ax3 = fig.add_subplot(2, 2, 3)
    colors = {'verytiny': '#e41a1c', 'tiny': '#ff7f00',
              'small': '#377eb8', 'medium': '#4daf4a'}
    by_label = defaultdict(list)
    for r in matched_preds:
        by_label[r['area_label']].append(r)

    for label in ['verytiny', 'tiny', 'small', 'medium']:
        preds = by_label.get(label, [])
        if len(preds) < 10:
            continue
        bc, bp, _, lece = _compute_calibration_curve(preds, n_bins=10)
        ax3.plot(bc, bp, 'o-', color=colors[label], markersize=5,
                 label=f'{label} (n={len(preds):,}, ECE={lece:.4f})',
                 linewidth=1.2)

    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax3.set_xlabel('Confidence score', fontsize=11)
    ax3.set_ylabel('Precision @IoU=0.5', fontsize=11)
    ax3.set_title('Reliability diagram by area range', fontsize=12)
    ax3.legend(fontsize=8, loc='upper left')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(-0.02, 1.05)
    ax3.grid(True, alpha=0.3)

    # -- Bottom-right: ECE and Precision summary table --
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    rows = [['Area range', '#Preds', '#TP', 'Precision', 'ECE']]
    for label in ['verytiny', 'tiny', 'small', 'medium']:
        s = per_area[label]
        rows.append([
            label,
            str(s['n_preds']),
            str(s['n_tp']),
            f"{s['precision']:.4f}" if not np.isnan(s['precision']) else 'N/A',
            f"{s['ece']:.4f}" if not np.isnan(s['ece']) else 'N/A',
        ])
    # Add 'all' row
    n_all = len(matched_preds)
    n_tp_all = sum(1 for r in matched_preds if r['is_tp'])
    rows.append([
        'ALL',
        str(n_all),
        str(n_tp_all),
        f"{n_tp_all / n_all:.4f}" if n_all > 0 else 'N/A',
        f"{ece:.4f}",
    ])

    table = ax4.table(cellText=rows, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    # Style header
    for j in range(5):
        table[0, j].set_facecolor('#40466e')
        table[0, j].set_text_props(color='white', fontweight='bold')
    # Highlight rows
    for i in range(1, len(rows)):
        for j in range(5):
            if i % 2 == 0:
                table[i, j].set_facecolor('#f0f0f0')
    ax4.set_title('Per-area-range calibration summary', fontsize=12, fontweight='bold')

    fig.suptitle(f'Calibration Analysis — IoU=0.5  {model_label}',
                 fontsize=14, fontweight='bold')
    _plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    _plt.close(fig)
    print(f'  Calibration plot saved → {out_path}')


def _plot_score_area(matched_preds: list, out_path: str, model_label: str = ''):
    """Save prediction score vs **resized** prediction area scatter plot.

    The x-axis uses *resized* (model-input) area so object sizes reflect
    what the model actually sees after the ``Resize`` transform.
    """
    _ensure_mpl()

    # Use resized areas for the x-axis — this is what the model "sees".
    scores = np.array([r['score'] for r in matched_preds], dtype=np.float64)
    areas = np.array([r['area_resized'] for r in matched_preds],
                     dtype=np.float64)
    is_tp = np.array([r['is_tp'] for r in matched_preds], dtype=bool)
    area_labels = np.array([r['area_label'] for r in matched_preds])

    tp_mask = is_tp
    fp_mask = ~is_tp

    fig = _plt.figure(figsize=(16, 10))

    # -- Top-left: scatter, log-log --
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(areas[tp_mask], scores[tp_mask], s=3, alpha=0.35,
                c='#2ca02c', label=f'TP  (n={tp_mask.sum():,})', edgecolors='none')
    ax1.scatter(areas[fp_mask], scores[fp_mask], s=3, alpha=0.25,
                c='#d62728', label=f'FP  (n={fp_mask.sum():,})', edgecolors='none')

    # Draw AITOD area-range boundaries (vertical lines)
    boundaries = [8**2, 16**2, 32**2]  # Note: these are original-space thresholds
    b_labels = ['verytiny/tiny', 'tiny/small', 'small/medium']
    for b, bl in zip(boundaries, b_labels):
        ax1.axvline(x=b, color='gray', linestyle=':', alpha=0.5)
        ax1.text(b * 0.95, 1.01, bl, fontsize=7, color='gray',
                 ha='right', transform=ax1.get_xaxis_transform())

    ax1.set_xscale('log')
    ax1.set_xlabel('Prediction area (resized px²) — log scale', fontsize=11)
    ax1.set_ylabel('Confidence score', fontsize=11)
    ax1.set_title('Score vs area (log-log)', fontsize=12)
    ax1.legend(fontsize=8, markerscale=3)
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=0.3)

    # -- Top-right: binned area vs avg score & precision --
    ax2 = fig.add_subplot(2, 2, 2)
    if len(areas) > 0:
        min_log = np.log10(max(areas.min(), 1e-2))
        max_log = np.log10(areas.max())
        bins = 10 ** np.linspace(min_log, max_log, 21)
    else:
        bins = np.linspace(0, 1, 21)

    bin_centers_area = np.sqrt(bins[:-1] * bins[1:])
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
    ax2.set_xlabel('Prediction area (resized px²) — log scale', fontsize=11)
    ax2.set_ylabel('Avg score / Precision', fontsize=11)
    ax2.set_title('Avg score & precision by area bin', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(True, alpha=0.3)
    for x, y, c in zip(bin_centers_area, avg_scores, bin_counts_area):
        if c > 0:
            ax2.text(x, y + 0.04, str(c), ha='center', fontsize=6.5,
                     color='gray', alpha=0.8)

    # -- Bottom-left: per-area-range score histograms --
    ax3 = fig.add_subplot(2, 2, 3)
    colors = {'verytiny': '#e41a1c', 'tiny': '#ff7f00',
              'small': '#377eb8', 'medium': '#4daf4a'}
    for i, label in enumerate(['verytiny', 'tiny', 'small', 'medium']):
        mask = area_labels == label
        if mask.sum() == 0:
            continue
        ax3.hist(scores[mask], bins=30, range=(0, 1), alpha=0.5,
                 color=colors[label], label=f'{label} (n={mask.sum():,})',
                 histtype='stepfilled', linewidth=0.5)

    ax3.set_xlabel('Confidence score', fontsize=11)
    ax3.set_ylabel('Number of predictions', fontsize=11)
    ax3.set_title('Score distribution by area range', fontsize=12)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # -- Bottom-right: per-area-range precision vs score threshold --
    ax4 = fig.add_subplot(2, 2, 4)
    by_label = defaultdict(list)
    for r in matched_preds:
        by_label[r['area_label']].append(r)

    for label in ['verytiny', 'tiny', 'small', 'medium']:
        preds = by_label.get(label, [])
        if len(preds) < 10:
            continue
        s = np.array([r['score'] for r in preds], dtype=np.float64)
        tp = np.array([r['is_tp'] for r in preds], dtype=np.float64)
        order = np.argsort(-s)
        s_desc = s[order]
        tp_cum = np.cumsum(tp[order])

        thrs = np.linspace(0.0, 1.0, 100)
        prec = np.zeros_like(thrs)
        for i, thr in enumerate(thrs):
            mask = s_desc >= thr
            n = mask.sum()
            if n > 0:
                prec[i] = tp_cum[mask].sum() / n

        ax4.plot(thrs, prec, color=colors[label], linewidth=1.5,
                 label=f'{label} (n={len(preds):,})')

    ax4.set_xlabel('Score threshold τ', fontsize=11)
    ax4.set_ylabel('Precision @IoU=0.5', fontsize=11)
    ax4.set_title('Cumulative precision by area range', fontsize=12)
    ax4.legend(fontsize=8)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(-0.02, 1.05)
    ax4.grid(True, alpha=0.3)

    fig.suptitle(f'Score–Area Analysis — IoU=0.5  {model_label}  '
                 f'(areas in resized / model-input px²)',
                 fontsize=14, fontweight='bold')
    _plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    _plt.close(fig)
    print(f'  Score-area plot saved   → {out_path}')


# ---------------------------------------------------------------------------
#  Comparison mode — load saved pickles and overlay
# ---------------------------------------------------------------------------

def _load_pickle_data(paths, labels=None):
    """Load one or more ``calibration_data.pkl`` files for comparison.

    Parameters
    ----------
    paths : list[str]
        Paths to ``.pkl`` files saved by ``main()``.
    labels : list[str] | None
        Display names (default: derived from ``model_name`` in the pickle).

    Returns
    -------
    list[tuple]  ``(matched_predictions, label_str, model_name)``
    """
    datasets = []
    for i, p in enumerate(paths):
        with open(p, 'rb') as f:
            obj = pickle.load(f)
        preds = obj['matched_predictions']
        name = obj.get('model_name', Path(p).parent.name)
        if labels and i < len(labels):
            lbl = labels[i]
        else:
            lbl = name
        datasets.append((preds, lbl, name))
        print(f"  Loaded {p}  →  {len(preds):,} predictions  label='{lbl}'")
    return datasets


def _plot_compare_score_area(datasets, save_path=None, area_key='resized',
                             title=None, figsize=(12, 8), show_fp=False):
    """Overlaid score vs prediction-area scatter for comparing ≥2 models.

    Parameters
    ----------
    datasets : list[tuple]
        ``(matched_predictions, label, model_name)`` from ``_load_pickle_data``.
    area_key : str
        ``'resized'`` (model-input px², default) or ``'original'``.
    show_fp : bool
        Whether to also plot false positives as faint 'x' markers.
    """
    _apply_pub_style()

    area_field = 'area_resized' if area_key == 'resized' else 'area_original'
    xlabel = ('Prediction Area (resized / model-input px²)' if area_key == 'resized'
              else 'Prediction Area (original image px²)')
    if title is None:
        title = 'Score vs Prediction Area Comparison'

    fig, ax = _plt.subplots(figsize=figsize)

    for i, (data, lbl, name) in enumerate(datasets):
        color = COLORS[i % len(COLORS)]

        tp_data = [d for d in data if d['is_tp']]
        fp_data = [d for d in data if not d['is_tp']]

        if tp_data:
            a = np.array([d[area_field] for d in tp_data])
            s = np.array([d['score'] for d in tp_data])
            ax.scatter(a, s, alpha=0.15, s=6, color=color, edgecolors='none',
                       label=f'{lbl}  TP (n={len(tp_data):,})')

        if show_fp and fp_data:
            fp_scores = np.array([d['score'] for d in fp_data])
            fp_areas = np.array([d[area_field] for d in fp_data])
            ax.scatter(fp_areas, fp_scores, alpha=0.08, s=6, color=color,
                       marker='x', label=f'{lbl}  FP (n={len(fp_data):,})')

    # Area-range boundary lines (original-space thresholds, labelled)
    boundaries = [8 ** 2, 16 ** 2, 32 ** 2]
    b_labels = ['verytiny ↔ tiny', 'tiny ↔ small', 'small ↔ medium']
    for b, bl in zip(boundaries, b_labels):
        ax.axvline(x=b, color='gray', linestyle=':', alpha=0.4)
        ax.text(b * 0.95, 1.01, bl, fontsize=7, color='gray',
                ha='right', transform=ax.get_xaxis_transform())

    ax.set_xscale('log')
    ax.set_xlim(*X_LIM_LOG)
    ax.set_ylim(*Y_LIM)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Confidence Score')
    ax.set_title(title)
    ax.legend(markerscale=2, loc='best')

    _plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f'  Compare score-area plot saved → {save_path}')

    _plt.close(fig)
    return fig


def _plot_compare_calibration(datasets, save_path=None, n_bins=10,
                              title=None, figsize=(15, 6)):
    """Overlaid calibration curves + grouped gap bars for ≥2 models.

    Left panel:  precision vs confidence with ECE/MCE in legend.
    Right panel:  grouped gap bars (precision − avg confidence) per bin.
    """
    _apply_pub_style()

    cals = []
    labels = []
    for data, lbl, name in datasets:
        cal = _compute_calibration_full(data, n_bins=n_bins)
        if cal is not None:
            cals.append(cal)
            labels.append(lbl)

    if not cals:
        print('  No calibration data for any dataset.')
        return None

    n_models = len(cals)
    fig, axes = _plt.subplots(1, 2, figsize=figsize)

    # ---- Left: overlaid calibration curves ----
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
    for i, (cal, lbl) in enumerate(zip(cals, labels)):
        color = COLORS[i % len(COLORS)]
        ax.plot(cal['bin_centers'], cal['precision'], '-', color=color,
                alpha=0.8, linewidth=1.5)
        sizes = np.maximum(
            cal['count'] / max(cal['count'].max(), 1) * 100, 25)
        ax.scatter(cal['bin_centers'], cal['precision'], s=sizes,
                   color=color, alpha=0.85, edgecolors='k', linewidths=0.5,
                   zorder=4,
                   label=f"{lbl}  ECE={cal['ece']:.4f}  MCE={cal['mce']:.4f}")

    ax.set_xlabel('Confidence')
    ax.set_ylabel('Precision @IoU=0.5')
    ax.set_title(title or 'Calibration Comparison')
    ax.legend(loc='upper left')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # ---- Right: grouped gap bars ----
    ax2 = axes[1]
    bar_width = 0.8 / n_models
    bin_edges = cals[0]['bin_edges']
    for i, (cal, lbl) in enumerate(zip(cals, labels)):
        color = COLORS[i % len(COLORS)]
        gap = cal['precision'] - cal['avg_conf']
        x = np.arange(n_bins) + (i - (n_models - 1) / 2) * bar_width
        ax2.bar(x, gap, width=bar_width, color=color, alpha=0.85,
                edgecolor='k', linewidth=0.5, label=lbl)
    ax2.axhline(0, color='k', linewidth=0.8)
    ax2.set_xlabel('Confidence Bin')
    ax2.set_ylabel('Precision − Confidence')
    ax2.set_title('Gap per Bin')
    ax2.set_xticks(range(n_bins))
    ax2.set_xticklabels(
        [f'{bin_edges[j]:.1f}' for j in range(n_bins)], fontsize=9)
    ax2.legend()

    _plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f'  Compare calibration plot saved → {save_path}')

    _plt.close(fig)
    return fig


def _save_compare_report(datasets, save_path, n_bins=10):
    """Save text report: overall ECE/MCE + per-size (verytiny/tiny/small/medium)
    breakdown + per-bin detail for each model.

    The report is also printed to stdout.
    """
    lines = []
    sep = '=' * 78
    sub = '─' * 78

    lines.append(sep)
    lines.append('  Calibration Comparison Report')
    lines.append(sep)

    # ---- Overall summary table ----
    lines.append('')
    lines.append(f"  {'Model':>28s}  {'TP':>8s}  {'FP':>8s}  {'ECE':>8s}  {'MCE':>8s}")
    lines.append(f"  {'─' * 68}")

    all_cals = []
    for data, lbl, name in datasets:
        tp = sum(1 for d in data if d['is_tp'])
        fp = sum(1 for d in data if not d['is_tp'])
        cal = _compute_calibration_full(data, n_bins=n_bins)
        all_cals.append((cal, lbl))
        ece = f"{cal['ece']:.4f}" if cal else 'N/A'
        mce = f"{cal['mce']:.4f}" if cal else 'N/A'
        lines.append(f"  {lbl:>28s}  {tp:>8,}  {fp:>8,}  {ece:>8s}  {mce:>8s}")

    # ---- Per-size breakdown per model ----
    for data, lbl, name in datasets:
        lines.append('')
        lines.append(sub)
        lines.append(f"  Per-size breakdown: {lbl}")
        lines.append(sub)
        lines.append(f"  {'Size':>12s}  {'#Preds':>8s}  {'#TP':>8s}  "
                     f"{'Prec':>8s}  {'ECE':>8s}  {'MCE':>8s}")
        lines.append(f"  {'─' * 58}")

        per_area = _per_area_calibration_full(data, n_bins=n_bins)
        for label in ['verytiny', 'tiny', 'small', 'medium']:
            s = per_area[label]
            prec_str = (f"{s['precision']:.4f}"
                        if not np.isnan(s['precision']) else 'N/A')
            ece_str = (f"{s['ece']:.4f}"
                       if not np.isnan(s['ece']) else 'N/A')
            mce_str = (f"{s['mce']:.4f}"
                       if not np.isnan(s['mce']) else 'N/A')
            lines.append(f"  {label:>12s}  {s['n_preds']:>8,}  "
                         f"{s['n_tp']:>8,}  {prec_str:>8s}  "
                         f"{ece_str:>8s}  {mce_str:>8s}")

        # Global row
        cal = _compute_calibration_full(data, n_bins=n_bins)
        if cal:
            n_all = len(data)
            n_tp_all = sum(1 for d in data if d['is_tp'])
            prec_all = n_tp_all / n_all if n_all > 0 else float('nan')
            prec_str = (f"{prec_all:.4f}"
                        if not np.isnan(prec_all) else 'N/A')
            lines.append(f"  {'─' * 58}")
            lines.append(f"  {'ALL':>12s}  {n_all:>8,}  {n_tp_all:>8,}  "
                         f"{prec_str:>8s}  {cal['ece']:>8.4f}  "
                         f"{cal['mce']:>8.4f}")

    # ---- Per-bin detail for each model ----
    for data, lbl, name in datasets:
        cal = _compute_calibration_full(data, n_bins=n_bins)
        if cal is None:
            continue
        lines.append('')
        lines.append(sub)
        lines.append(f"  Per-bin detail: {lbl}")
        lines.append(sub)
        lines.append(f"  {'Bin':>14s}  {'Count':>7s}  {'AvgConf':>8s}  "
                     f"{'Prec':>8s}  {'Gap':>8s}")
        lines.append(f"  {'─' * 51}")
        for i in range(n_bins):
            if cal['count'][i] > 0:
                gap = cal['precision'][i] - cal['avg_conf'][i]
                lines.append(
                    f"  [{cal['bin_edges'][i]:.1f},"
                    f"{cal['bin_edges'][i + 1]:.1f})  "
                    f"{cal['count'][i]:>7d}  {cal['avg_conf'][i]:>8.4f}  "
                    f"{cal['precision'][i]:>8.4f}  {gap:>+8.4f}")

    lines.append('')
    lines.append(sep)
    lines.append('  End of report')
    lines.append(sep)

    report = '\n'.join(lines)

    # Write to file
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report)

    # Also print to stdout
    print('\n' + report)
    print(f'\n  Report saved → {save_path}')


def _run_compare_mode(args):
    """Entry point for ``--load``: compare multiple experiments."""
    print(f"\n{'='*60}")
    print('Compare mode: loading saved pickle files')
    print(f"{'='*60}")

    datasets = _load_pickle_data(args.load, labels=args.labels)

    # Output directory
    out_dir = Path(args.save_dir) if args.save_dir else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    area_labels = {'resized': 'model-input px²', 'original': 'original px²'}
    area_str = area_labels.get(args.area_key, args.area_key)
    print(f'\n  Area key         : {args.area_key} ({area_str})')
    print(f'  Calibration bins : {args.n_bins}')
    print(f'  Output directory : {out_dir}')

    # ---- Report ----
    report_path = out_dir / 'comparison_report.txt'
    _save_compare_report(datasets, str(report_path), n_bins=args.n_bins)

    # ---- Plots ----
    cal_path = out_dir / 'compare_calibration.png'
    _plot_compare_calibration(datasets, save_path=str(cal_path),
                              n_bins=args.n_bins)

    area_path = out_dir / 'compare_score_vs_area.png'
    _plot_compare_score_area(datasets, save_path=str(area_path),
                             area_key=args.area_key, show_fp=args.show_fp)

    print(f"\n{'='*60}")
    print(f'All comparison outputs in: {out_dir}/')
    print(f'  comparison_report.txt')
    print(f'  compare_calibration.png')
    print(f'  compare_score_vs_area.png')
    print(f"{'='*60}")


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
        default=None,
        help='Path to the .pth checkpoint to evaluate '
             '(required for eval mode, ignored with --load)')
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

    # ---- Compare mode (load saved pickles, skip eval) ----
    parser.add_argument(
        '--load', type=str, nargs='+', default=None,
        help='Load one or more calibration_data.pkl files for comparison '
             '(skips evaluation entirely)')
    parser.add_argument(
        '--labels', type=str, nargs='+', default=None,
        help='Display names for each loaded pickle (same order as --load). '
             'Default: derived from model_name in the pickle.')
    parser.add_argument(
        '--area-key', type=str, default='resized',
        choices=['resized', 'original'],
        help='Which area metric for the x-axis of scatter plots '
             '(default: resized = model-input px²)')
    parser.add_argument(
        '--n-bins', type=int, default=10,
        help='Number of confidence bins for calibration (default: 10)')
    parser.add_argument(
        '--show-fp', action='store_true', default=False,
        help='Also show false positives on the compare score-area scatter')
    parser.add_argument(
        '--save-dir', type=str, default=None,
        help='Directory for comparison outputs '
             '(default: --out-dir, or eval_outputs/compare)')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Compare mode: load saved pickles and overlay — no eval needed
    # ------------------------------------------------------------------
    if args.load:
        _run_compare_mode(args)
        return

    if not args.checkpoint:
        raise SystemExit(
            'Error: --checkpoint is required for eval mode '
            '(or use --load for compare mode).')

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
        proposal_nums=(1, 100, 300),
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

    with open(pred_json_path) as f:
        raw_predictions = json.load(f)

    if len(raw_predictions) == 0:
        print("\n[WARN] Zero predictions — skipping calibration analysis.")
        return

    coco_gt = COCO(ann_file)
    cat_id_to_name = {cat['id']: cat['name']
                      for cat in coco_gt.loadCats(coco_gt.getCatIds())}

    # Build per-image resize-scale map so we can convert areas from
    # original-image px² → model-input (resized) px².
    max_size = _extract_resize_max_size(cfg)
    scale_map = _build_scale_map(coco_gt, max_size)
    print(f"\n  Resize max size: {max_size} px  (keep_ratio=True)")
    print(f"  Unique scales:   {sorted(set(f'{v:.4f}' for v in scale_map.values()))}")

    # ------------------------------------------------------------------
    # 5. Match predictions to GT (greedy, per image×category, IoU=0.5)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Calibration analysis: matching predictions to GT  (IoU=0.5)")
    print(f"{'='*60}")

    print(f"  Predictions : {len(raw_predictions):,}")
    print(f"  GT boxes    : {len(coco_gt.loadAnns(coco_gt.getAnnIds())):,}")

    matched = _greedy_match(coco_gt, raw_predictions, scale_map, iou_thr=0.5)

    n_tp = sum(1 for r in matched if r['is_tp'])
    n_fp = len(matched) - n_tp
    print(f"  TP          : {n_tp:,}")
    print(f"  FP          : {n_fp:,}")
    print(f"  Precision   : {n_tp / len(matched):.4f}  (at IoU=0.5, all scores)")

    # ------------------------------------------------------------------
    # 5b. Per-area-range breakdown (resized-area based)
    # ------------------------------------------------------------------
    print(f"\n  --- Per-area-range breakdown (resized area) ---")
    per_area = _per_area_calibration(matched)
    for label in ['verytiny', 'tiny', 'small', 'medium']:
        s = per_area[label]
        prec_str = f"{s['precision']:.4f}" if not np.isnan(s['precision']) else 'N/A'
        ece_str = f"{s['ece']:.4f}" if not np.isnan(s['ece']) else 'N/A'
        print(f"    {label:>10s}:  preds={s['n_preds']:>6,}  "
              f"TP={s['n_tp']:>6,}  precision={prec_str}  ECE={ece_str}")

    # ------------------------------------------------------------------
    # 6. Save intermediate data for later reuse
    # ------------------------------------------------------------------
    pkl_path = os.path.join(args.out_dir, 'calibration_data.pkl')
    pkl_data = {
        'model_name': os.path.splitext(os.path.basename(args.checkpoint))[0],
        'config': args.config,
        'checkpoint': args.checkpoint,
        'matched_predictions': matched,
        'cat_id_to_name': cat_id_to_name,
        'iou_thr': 0.5,
        'resize_max_size': max_size,
        'scale_map': scale_map,
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
    print(f"  - {os.path.basename(pred_json_path)}  (COCO-format predictions)")
    print(f"  - calibration_data.pkl                 (matched preds, reusable)")
    print(f"  - calibration_score_precision.png      (score-precision + per-area)")
    print(f"  - score_vs_area.png                    (score-area scatter)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
