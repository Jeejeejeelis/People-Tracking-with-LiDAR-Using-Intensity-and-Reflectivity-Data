import numpy as np
import os
from pathlib import Path
import logging
from core.person_tracker import PersonTracker
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from pcdet.config import cfg, cfg_from_yaml_file

def create_logger():
    logger = logging.getLogger('pfsd_debug')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)
    return logger

def run_pfsd_dbscan(scene_id=None, num_samples=None):
    logger = create_logger()
    
    # Load dataset config
    cfg_file = "OpenPCDet/tools/cfgs/dataset_configs/nuscenes_dataset.yaml"
    cfg_from_yaml_file(cfg_file, cfg)
    
    # PFSD structure:
    # OpenPCDet/data/nuscenes/v1.0-mini/
    #   pfsd_infos_10sweeps_val.pkl
    #   v1.0-mini/ (json files)
    
    cfg.DATA_PATH = 'OpenPCDet/data/nuscenes/v1.0-mini'
    cfg.VERSION = '' # Avoid subfolder append
    
    base_root = Path(cfg.DATA_PATH)
    info_path = base_root / 'pfsd_infos_10sweeps_val.pkl'
    print(f"Checking info file: {info_path.absolute()} - Exists: {info_path.exists()}")
    
    cfg.INFO_PATH['test'] = ['pfsd_infos_10sweeps_val.pkl']
    
    dataset = NuScenesDataset(
        dataset_cfg=cfg,
        class_names=['pedestrian'],
        training=False,
        root_path=base_root,
        logger=logger
    )

    # Load token to scene mapping
    token_to_scene = {}
    with open('core/token_to_scene.tsv', 'r') as f:
        for line in f:
            token, scene = line.strip().split('\t')
            token_to_scene[token] = scene

    tracker = PersonTracker()
    
    # Group samples by scene
    scene_samples = {}
    for i in range(len(dataset)):
        info = dataset.infos[i]
        token = info['token']
        scene_token = token_to_scene.get(token, 'unknown')
        if scene_token not in scene_samples:
            scene_samples[scene_token] = []
        scene_samples[scene_token].append(i)
    
    for scene_token in scene_samples:
        scene_samples[scene_token].sort(key=lambda idx: dataset.infos[idx]['timestamp'])

    scene_tokens = list(scene_samples.keys())
    if scene_id is not None and int(scene_id) < len(scene_tokens):
        target_scenes = [scene_tokens[int(scene_id)]]
    else:
        target_scenes = scene_tokens[:1]

    for scene_token in target_scenes:
        print(f"\n--- Processing Scene {scene_token} ---")
        tracker = PersonTracker()
        
        indices = scene_samples[scene_token]
        if num_samples is not None:
            indices = indices[:num_samples]
            
        for i, idx in enumerate(indices):
            data_dict = dataset[idx]
            pc = data_dict['points'] # (N, 5) [x, y, z, intensity, timestamp]
            
            # PersonTracker expects [x, y, z, intensity]
            pc_4 = pc[:, :4]
            
            filtered_pc = tracker.preprocess(pc_4)
            clusters = tracker.cluster(filtered_pc)
            detections = tracker.filter_pedestrians(clusters)
            tracked_peds = tracker.track(detections)
            
            print(f"Frame {i:03d} | Detections: {len(detections):3d} | Active Tracks: {len(tracked_peds):3d}")

    print("\nPFSD DBSCAN verification complete.")

if __name__ == "__main__":
    import sys
    sid = sys.argv[1] if len(sys.argv) > 1 else None
    ns = int(sys.argv[2]) if len(sys.argv) > 2 else 20 # Run 20 frames by default
    run_pfsd_dbscan(scene_id=sid, num_samples=ns)
