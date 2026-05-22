import argparse
import os
import re
from datetime import datetime
import pandas as pd

def find_latest_exp(exp_folder):
    exp_ids = [d for d in os.listdir(exp_folder) if re.match(r'^\d{8}_\d{6}$', d)]
    exp_ids.sort(key=lambda x: datetime.strptime(x, '%Y%m%d_%H%M%S'), reverse=True)
    return os.path.join(exp_folder, exp_ids[0]) if exp_ids else None

def parse_log_file(log_path):
    pattern = re.compile(r'Epoch\(val\)\s+\[(\d+)\].*?coco/bbox_mAP: (\d+\.\d{4})')
    results = []
    
    with open(log_path, 'r') as f:
        for line in f:
            if 'Epoch(val)' in line and 'coco/bbox_mAP' in line:
                match = pattern.search(line)
                if match:
                    epoch = int(match.group(1))
                    metrics = {'epoch': epoch}
                    metrics.update({k: float(v) for k, v in re.findall(r'(\S+): (\d+\.\d+)', line)})
                    results.append(metrics)
    return pd.DataFrame(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_folder', type=str, required=True, help='实验文件夹路径')
    parser.add_argument('--exp_id', type=str, help='指定实验ID（格式: YYYYMMDD_HHMMSS）')
    args = parser.parse_args()

    # 确定实验路径
    if args.exp_id:
        exp_path = os.path.join(args.exp_folder, args.exp_id)
    else:
        exp_path = find_latest_exp(args.exp_folder)
    
    log_file = os.path.join(exp_path, f'{os.path.basename(exp_path)}.log')
    
    # 解析并展示结果
    df = parse_log_file(log_file)
    print(f'\nParse Result ({os.path.basename(exp_path)}):\n')
    print(df.sort_values('epoch').to_string(index=False, float_format='%.4f'))