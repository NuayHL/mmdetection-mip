import torch
from operator import itemgetter

def list_slice(slice_idx, target_list):
    """do not use itemgetter here"""
    if isinstance(slice_idx, torch.Tensor):
        slice_idx = slice_idx.tolist()
    return [target_list[idx] for idx in slice_idx]

def dict_sum_up(ori_dict, new_dict):
    for k, v in new_dict.items():
        if k in ori_dict.keys():
            ori_dict[k] += v
        else:
            ori_dict[k] = v
    return ori_dict
