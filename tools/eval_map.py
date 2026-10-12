import pickle
import os
import numpy as np
import sys
import logging
from pathlib import Path

# Change cwd to PFSD/tools to satisfy OpenPCDet's relative config paths
original_cwd = os.getcwd()
os.chdir('PFSD/tools')
sys.path.append(os.path.join(os.getcwd(), '..'))

from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from pcdet.config import cfg, cfg_from_yaml_file

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

pkl_files = [
    '../../results/raw_detections_cp_swap.pkl'
]

cfg_from_yaml_file('../../configs/pfsd_hmfn.yaml', cfg)
cfg.DATA_CONFIG.DATA_PATH = '../data/nuscenes'
cfg.DATA_CONFIG.VERSION = 'v1.0-mini'

dataset = NuScenesDataset(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=cfg.CLASS_NAMES,
    training=False,
    root_path=Path('../data/nuscenes'),
    logger=logger
)

for pkl in pkl_files:
    if not os.path.exists(pkl): continue
    with open(pkl, 'rb') as f:
        det = pickle.load(f)
    
    # Add 'name' key and ensure it's mapped to NuScenes classes
    for d in det:
        d['name'] = np.array(['pedestrian'] * len(d['boxes_lidar']))
    
    out_dir = pkl.replace('.pkl', '_eval')
    os.makedirs(out_dir, exist_ok=True)
    try:
        res_dict, _ = dataset.evaluation(det, ['pedestrian'], output_path=out_dir)
        print(f"File: {pkl}")
        print(f"Pedestrian mAP: {res_dict['mean_dist_aps']['pedestrian']:.4f}")
        print(f"AP@0.5: {res_dict['label_aps']['pedestrian']['0.5']:.4f}")
        print(f"AP@1.0: {res_dict['label_aps']['pedestrian']['1.0']:.4f}")
        print(f"AP@2.0: {res_dict['label_aps']['pedestrian']['2.0']:.4f}")
        print(f"AP@4.0: {res_dict['label_aps']['pedestrian']['4.0']:.4f}")
    except Exception as e:
        print(f"Failed {pkl}: {e}")
