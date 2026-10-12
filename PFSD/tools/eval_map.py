import pickle
import os
import numpy as np
import sys
from pathlib import Path
sys.path.append(os.path.join(os.getcwd(), '../'))

from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from pcdet.config import cfg, cfg_from_yaml_file

pkl_files = [
    '../../results/raw_detections_hmfn_baseline.pkl',
    '../../results/raw_detections_hmfn_kitchen_sink.pkl',
    '../../results/raw_detections_vn_baseline.pkl',
    '../../results/raw_detections_vn_decoupled.pkl',
    '../../results/raw_detections_vn_kitchen_sink.pkl'
]

cfg_from_yaml_file('../../configs/pfsd_hmfn.yaml', cfg)
cfg.DATA_CONFIG.DATA_PATH = '../data/nuscenes'
cfg.DATA_CONFIG.VERSION = 'v1.0-mini'

dataset = NuScenesDataset(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=cfg.CLASS_NAMES,
    training=False,
    root_path=Path('../data/nuscenes'),
    logger=__import__('logging').getLogger('eval')
)

for pkl in pkl_files:
    if not os.path.exists(pkl): continue
    with open(pkl, 'rb') as f:
        det = pickle.load(f)
    
    for d in det:
        d['name'] = np.array(['pedestrian'] * len(d['boxes_lidar']))
    
    out_dir = pkl.replace('.pkl', '_eval')
    os.makedirs(out_dir, exist_ok=True)
    try:
        if not os.path.exists(os.path.join(out_dir, 'metrics_summary.json')):
            res_dict, _ = dataset.evaluation(det, ['pedestrian'], output_path=out_dir)
        import json
        with open(os.path.join(out_dir, 'metrics_summary.json')) as f:
            data = json.load(f)
            mean_ap = data['mean_dist_aps']['pedestrian'] * 100
            ap_05 = data['label_aps']['pedestrian']['0.5'] * 100
            ap_10 = data['label_aps']['pedestrian']['1.0'] * 100
            ap_20 = data['label_aps']['pedestrian']['2.0'] * 100
            ap_40 = data['label_aps']['pedestrian']['4.0'] * 100
            print(f"{pkl}: {mean_ap:.2f} | {ap_05:.2f} | {ap_10:.2f} | {ap_20:.2f} | {ap_40:.2f}")
    except Exception as e:
        print(f"Failed {pkl}: {e}")
