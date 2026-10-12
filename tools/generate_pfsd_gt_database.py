import numpy as np
import os
import pickle
from pathlib import Path
import logging
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils

def generate_database():
    logger = common_utils.create_logger()
    
    # Load PFSD config
    cfg_file = "OpenPCDet/tools/cfgs/dataset_configs/pfsd_dataset.yaml"
    cfg_from_yaml_file(cfg_file, cfg)
    
    # Correct paths for PFSD
    # Data is at: OpenPCDet/data/nuscenes/v1.0-mini/
    # infos are at: OpenPCDet/data/nuscenes/v1.0-mini/pfsd_infos_10sweeps_val.pkl
    
    cfg.DATA_PATH = 'OpenPCDet/data/nuscenes/v1.0-mini'
    cfg.VERSION = '' # Stop it from adding extra subfolders
    cfg.INFO_PATH['train'] = ['pfsd_infos_10sweeps_val.pkl'] # Use val for now to verify logic works
    
    # Disable augmentation and resampling while creating database
    cfg.DATA_AUGMENTOR.DISABLE_AUG_LIST = ['gt_sampling']
    cfg.BALANCED_RESAMPLING = False
    
    available_classes = ['car', 'pedestrian', 'bus', 'motorcycle', 'barrier']

    dataset = NuScenesDataset(
        dataset_cfg=cfg,
        class_names=available_classes,
        training=True,
        root_path=Path(cfg.DATA_PATH),
        logger=logger
    )
    
    print(f"Dataset loaded with {len(dataset)} samples.")
    
    # Create the GT database
    # This will use get_lidar_with_sweeps, which we patched to include reflectivity!
    print("Generating GT database (including reflectivity)...")
    dataset.create_groundtruth_database(max_sweeps=10, used_classes=['pedestrian'])
    
    print("\nGT Database generation complete.")

if __name__ == "__main__":
    generate_database()
