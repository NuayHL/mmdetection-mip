# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import os.path as osp
import tempfile
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from mmengine.fileio import get_local_path, load
from mmengine.logging import MMLogger
from terminaltables import AsciiTable

from mmdet.datasets.api_wrappers import COCO, COCOeval, COCOevalMP
from mmdet.registry import METRICS
from .coco_metric import CocoMetric


@METRICS.register_module()
class VisDroneCocoMetric(CocoMetric):
    """VisDrone COCO evaluation metric.

    This metric extends CocoMetric with VisDrone-specific area ranges:
    - Area ranges: all, tiny (0-16²), small (16²-32²), medium (32²-64²),
      large (64²-1e5²)
    - MaxDets: [1, 100, 1500]

    Args:
        ann_file (str, optional): Path to the coco format annotation file.
        metric (str | List[str]): Metrics to be evaluated. Defaults to 'bbox'.
        classwise (bool): Whether to evaluate the metric class-wise.
            Defaults to False.
        proposal_nums (Sequence[int]): Numbers of proposals to be evaluated.
            Defaults to (1, 100, 1500) for VisDrone.
        iou_thrs (float | List[float], optional): IoU threshold to compute AP
            and AR. If not specified, IoUs from 0.5 to 0.95 will be used.
            Defaults to None.
        metric_items (List[str], optional): Metric result names to be
            recorded in the evaluation result. Defaults to None.
        format_only (bool): Format the output results without perform
            evaluation. Defaults to False.
        outfile_prefix (str, optional): The prefix of json files.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            Defaults to None.
        use_mp_eval (bool): Whether to use mul-processing evaluation.
    """
    default_prefix: Optional[str] = 'visdrone'

    def __init__(self,
                 ann_file: Optional[str] = None,
                 metric: Union[str, List[str]] = 'bbox',
                 classwise: bool = False,
                 proposal_nums: Sequence[int] = (1, 100, 1500),
                 iou_thrs: Optional[Union[float, Sequence[float]]] = None,
                 metric_items: Optional[Sequence[str]] = None,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 backend_args: dict = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 use_mp_eval: bool = False) -> None:
        # VisDrone 默认使用 (1, 100, 1500) 作为 proposal_nums
        super().__init__(
            ann_file=ann_file,
            metric=metric,
            classwise=classwise,
            proposal_nums=proposal_nums,
            iou_thrs=iou_thrs,
            metric_items=metric_items,
            format_only=format_only,
            outfile_prefix=outfile_prefix,
            backend_args=backend_args,
            collect_device=collect_device,
            prefix=prefix,
            use_mp_eval=use_mp_eval)

        # VisDrone 的 area ranges
        self.visdrone_area_rng = [
            [0 ** 2, 1e5 ** 2],      # all
            [0 ** 2, 16 ** 2],       # tiny (0 to 16²)
            [16 ** 2, 32 ** 2],      # small
            [32 ** 2, 64 ** 2],      # medium
            [64 ** 2, 1e5 ** 2],     # large
        ]
        self.visdrone_area_lbl = ['all', 'tiny', 'small', 'medium', 'large']

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results with VisDrone area ranges.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # split gt and prediction list
        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        if self._coco_api is None:
            # use converted gt json file to initialize coco api
            logger.info('Converting ground truth to coco format...')
            coco_json_path = self.gt_to_coco_json(
                gt_dicts=gts, outfile_prefix=outfile_prefix)
            self._coco_api = COCO(coco_json_path)

        # handle lazy init
        if self.cat_ids is None:
            self.cat_ids = self._coco_api.get_cat_ids(
                cat_names=self.dataset_meta['classes'])
        if self.img_ids is None:
            self.img_ids = self._coco_api.get_img_ids()

        # convert predictions to coco format and dump to json file
        result_files = self.results2json(preds, outfile_prefix)

        eval_results = OrderedDict()
        if self.format_only:
            logger.info('results are saved in '
                        f'{osp.dirname(outfile_prefix)}')
            return eval_results

        for metric in self.metrics:
            logger.info(f'Evaluating {metric}...')

            # TODO: May refactor fast_eval_recall to an independent metric?
            # fast eval recall
            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    preds, self.proposal_nums, self.iou_thrs, logger=logger)
                log_msg = []
                for i, num in enumerate(self.proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                logger.info(log_msg)
                continue

            # evaluate proposal, bbox and segm
            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = load(result_files[metric])
                predictions = _sanitize_predictions(predictions, logger)
                if iou_type == 'segm':
                    # Remove bbox for mask evaluation
                    for x in predictions:
                        x.pop('bbox')
                coco_dt = self._coco_api.loadRes(predictions)

            except IndexError:
                logger.error(
                    'The testing results of the whole dataset is empty.')
                break

            if self.use_mp_eval:
                coco_eval = COCOevalMP(self._coco_api, coco_dt, iou_type)
            else:
                coco_eval = COCOeval(self._coco_api, coco_dt, iou_type)

            # Set VisDrone-specific parameters
            coco_eval.params.catIds = self.cat_ids
            coco_eval.params.imgIds = self.img_ids
            coco_eval.params.maxDets = list(self.proposal_nums)
            coco_eval.params.iouThrs = self.iou_thrs
            # Set VisDrone area ranges
            coco_eval.params.areaRng = self.visdrone_area_rng
            coco_eval.params.areaRngLbl = self.visdrone_area_lbl

            # Define helper function to compute VisDrone metrics
            def _compute_visdrone_stats(coco_eval, print_summary=False):
                """Compute VisDrone-specific stats from COCOeval results."""
                def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100, print_result=False):
                    p = coco_eval.params
                    iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
                    titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
                    typeStr = '(AP)' if ap == 1 else '(AR)'
                    iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                        if iouThr is None else '{:0.2f}'.format(iouThr)

                    aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
                    mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
                    if ap == 1:
                        s = coco_eval.eval['precision']
                        if iouThr is not None:
                            t = np.where(iouThr == p.iouThrs)[0]
                            s = s[t]
                        s = s[:, :, :, aind, mind]
                    else:
                        s = coco_eval.eval['recall']
                        if iouThr is not None:
                            t = np.where(iouThr == p.iouThrs)[0]
                            s = s[t]
                        s = s[:, :, aind, mind]
                    if len(s[s > -1]) == 0:
                        mean_s = -1
                    else:
                        mean_s = np.mean(s[s > -1])
                    
                    if print_result:
                        logger.info(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
                    return mean_s

                # Compute stats: [mAP, mAP50, mAP75, AP_tiny, AP_small, AP_medium, AP_large,
                #                 AR@1, AR@100, AR@1500, AR_tiny, AR_small, AR_medium, AR_large]
                stats = np.zeros(14,)
                
                if print_summary:
                    logger.info('')
                    logger.info('=' * 80)
                    logger.info('VisDrone Evaluation Results:')
                    logger.info('=' * 80)
                
                # AP metrics
                stats[0] = _summarize(1, print_result=print_summary)  # mAP @[0.50:0.95]
                stats[1] = _summarize(1, iouThr=.5, maxDets=self.proposal_nums[2], print_result=print_summary)  # mAP50
                stats[2] = _summarize(1, iouThr=.75, maxDets=self.proposal_nums[2], print_result=print_summary)  # mAP75
                stats[3] = _summarize(1, areaRng='tiny', maxDets=self.proposal_nums[2], print_result=print_summary)
                stats[4] = _summarize(1, areaRng='small', maxDets=self.proposal_nums[2], print_result=print_summary)
                stats[5] = _summarize(1, areaRng='medium', maxDets=self.proposal_nums[2], print_result=print_summary)
                stats[6] = _summarize(1, areaRng='large', maxDets=self.proposal_nums[2], print_result=print_summary)
                
                # AR metrics
                stats[7] = _summarize(0, maxDets=self.proposal_nums[0], print_result=print_summary)  # AR@1
                stats[8] = _summarize(0, maxDets=self.proposal_nums[1], print_result=print_summary)  # AR@100
                stats[9] = _summarize(0, maxDets=self.proposal_nums[2], print_result=print_summary)  # AR@1500
                stats[10] = _summarize(0, areaRng='tiny', maxDets=self.proposal_nums[2], print_result=print_summary)
                stats[11] = _summarize(0, areaRng='small', maxDets=self.proposal_nums[2], print_result=print_summary)
                stats[12] = _summarize(0, areaRng='medium', maxDets=self.proposal_nums[2], print_result=print_summary)
                stats[13] = _summarize(0, areaRng='large', maxDets=self.proposal_nums[2], print_result=print_summary)
                
                if print_summary:
                    logger.info('=' * 80)
                    logger.info('')
                
                return stats

            # VisDrone metric names mapping
            visdrone_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_tiny': 3,
                'mAP_small': 4,
                'mAP_medium': 5,
                'mAP_large': 6,
                'AR@1': 7,
                'AR@100': 8,
                'AR@1500': 9,
                'AR_tiny@1500': 10,
                'AR_small@1500': 11,
                'AR_medium@1500': 12,
                'AR_large@1500': 13,
            }
            metric_items = self.metric_items
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in visdrone_metric_names:
                        raise KeyError(
                            f'metric item "{metric_item}" is not supported')

            if metric == 'proposal':
                coco_eval.params.useCats = 0
                coco_eval.evaluate()
                coco_eval.accumulate()
                # Compute VisDrone stats manually with summary printing
                visdrone_stats = _compute_visdrone_stats(coco_eval, print_summary=True)
                if metric_items is None:
                    metric_items = [
                        'AR@1', 'AR@100', 'AR@1500', 'AR_tiny@1500',
                        'AR_small@1500', 'AR_medium@1500', 'AR_large@1500'
                    ]

                for item in metric_items:
                    val = float(f'{visdrone_stats[visdrone_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                coco_eval.evaluate()
                coco_eval.accumulate()
                # Compute VisDrone stats manually with summary printing
                visdrone_stats = _compute_visdrone_stats(coco_eval, print_summary=True)

                if self.classwise:  # Compute per-category AP
                    precisions = coco_eval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, cat_id in enumerate(self.cat_ids):
                        t = []
                        # area range index 0: all area ranges
                        # max dets index -1: typically 1500 per image for VisDrone
                        nm = self._coco_api.loadCats(cat_id)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        t.append(f'{nm["name"]}')
                        t.append(f'{round(ap, 3)}')
                        eval_results[f'{nm["name"]}_precision'] = round(ap, 3)

                        # indexes of IoU @50 and @75
                        for iou in [0, 5]:
                            precision = precisions[iou, :, idx, 0, -1]
                            precision = precision[precision > -1]
                            if precision.size:
                                ap = np.mean(precision)
                            else:
                                ap = float('nan')
                            t.append(f'{round(ap, 3)}')

                        # indexes of area ranges: tiny, small, medium, large
                        for area in [1, 2, 3, 4]:
                            if area < len(self.visdrone_area_rng):
                                precision = precisions[:, :, idx, area, -1]
                                precision = precision[precision > -1]
                                if precision.size:
                                    ap = np.mean(precision)
                                else:
                                    ap = float('nan')
                                t.append(f'{round(ap, 3)}')
                            else:
                                t.append('nan')
                        results_per_category.append(tuple(t))

                    num_columns = len(results_per_category[0])
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = [
                        'category', 'mAP', 'mAP_50', 'mAP_75', 
                        'mAP_tiny', 'mAP_small', 'mAP_medium', 'mAP_large'
                    ]
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    logger.info('\n' + table.table)

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_tiny', 
                        'mAP_small', 'mAP_medium', 'mAP_large'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = visdrone_stats[visdrone_metric_names[metric_item]]
                    eval_results[key] = float(f'{round(val, 3)}')

                # Log VisDrone-specific metrics
                ap = visdrone_stats[:7]  # mAP, mAP50, mAP75, and 4 area ranges
                logger.info(f'{metric}_mAP_copypaste: {ap[0]:.3f} '
                            f'{ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                            f'{ap[4]:.3f} {ap[5]:.3f} {ap[6]:.3f}')

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

def _sanitize_predictions(predictions, logger):
    valid_preds = []
    for i, p in enumerate(predictions):
        bbox = p.get('bbox', None)
        score = p.get('score', None)
        cat_id = p.get('category_id', None)

        if bbox is None or score is None or cat_id is None:
            continue

        x, y, w, h = bbox

        # 基本合法性
        if not np.isfinite([x, y, w, h, score]).all():
            logger.warning(f'Drop NaN/Inf bbox: {bbox}')
            continue

        if w <= 0 or h <= 0:
            logger.warning(f'Drop invalid bbox (w/h<=0): {bbox}')
            continue

        if score < 0 or score > 1:
            logger.warning(f'Drop invalid score: {score}')
            continue

        valid_preds.append(p)

    if len(valid_preds) == 0:
        logger.error('All predictions filtered out!')

    return valid_preds

