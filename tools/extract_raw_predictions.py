import numpy as np
import os
from pathlib import Path
import torch
import logging
from core.dl_tracker import OpenPCDetTracker
from core.person_tracker import PersonTracker
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from pcdet.config import cfg, cfg_from_yaml_file
from tools.pfsd_evaluator import PFSDEvaluator
import json

def create_logger():
    logger = logging.getLogger('evaluate_pfsd_tracking')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)
    return logger

class PFSDTrackerWrapper(OpenPCDetTracker):
    def __init__(self, config_path, weights_path, tracker_type='dl'):
        super().__init__(config_path, weights_path, intensity_mode='raw')
        from pcdet.config import cfg
        try:
            self.ped_label = cfg.CLASS_NAMES.index('pedestrian') + 1
        except ValueError:
            self.ped_label = 9
        self.tracker_type = tracker_type

    def run_detection(self, points):
        if not self.model: return []
        
        # Determine number of features from model config
        num_features = self.model.dataset.dataset_cfg.get('POINT_FEATURE_ENCODING', {}).get('src_feature_list', [])
        expected_len = len(num_features)
        
        if points.shape[1] < expected_len:
            fill = np.zeros((points.shape[0], expected_len - points.shape[1]), dtype=points.dtype)
            points = np.hstack((points, fill))
        elif points.shape[1] > expected_len:
            points = points[:, :expected_len]
            
        # Frame-wide Reflectivity Preprocessing (Log, Winsorize, Z-score)
        dist_sq = np.sum(points[:, :3]**2, axis=1)
        frame_refl = np.log1p(points[:, 3] * dist_sq)
        if len(frame_refl) > 0:
            r_max_clip = np.percentile(frame_refl, 99)
            frame_refl = np.clip(frame_refl, a_min=None, a_max=r_max_clip)
            r_mean, r_std = frame_refl.mean(), frame_refl.std()
            if r_std > 1e-6:
                frame_refl = (frame_refl - r_mean) / r_std
        
        # Append preprocessed reflectivity to the end of the points array (for tracker extraction later)
        points_with_refl = np.hstack((points, frame_refl.reshape(-1, 1)))
        
        with torch.no_grad():
            input_dict = {
                'points': points,
                'frame_id': np.array(['000000']),
            }
            data_dict = self.model.dataset.prepare_data(data_dict=input_dict)
            data_dict = self.model.dataset.collate_batch([data_dict])
            from pcdet.models import load_data_to_gpu
            load_data_to_gpu(data_dict)
            
            pred_dicts, _ = self.model.forward(data_dict)
            pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()
            pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()

        pedestrians = []
        for i in range(len(pred_boxes)):
            if pred_labels[i] == self.ped_label and pred_scores[i] > 0.05:
                box = pred_boxes[i]
                center = box[:3]
                dims = box[3:6]
                p_min = center - (dims / 2)
                p_max = center + (dims / 2)
                
                # Frontier 1: Reflectivity Signature Extraction
                # Find raw LiDAR points inside this bounding box
                mask = (points_with_refl[:, 0] >= p_min[0]) & (points_with_refl[:, 0] <= p_max[0]) & \
                       (points_with_refl[:, 1] >= p_min[1]) & (points_with_refl[:, 1] <= p_max[1]) & \
                       (points_with_refl[:, 2] >= p_min[2]) & (points_with_refl[:, 2] <= p_max[2])
                
                points_in_box = points_with_refl[mask]
                if len(points_in_box) > 0:
                    reflectivity_signature = float(np.mean(points_in_box[:, -1]))
                    median_reflectivity = float(np.median(points_in_box[:, -1]))
                else:
                    reflectivity_signature = 0.0
                    median_reflectivity = 0.0
                
                pedestrians.append({
                    'center': center,
                    'bbox': (p_min, p_max),
                    'score': pred_scores[i],
                    'raw_box': pred_boxes[i],
                    'reflectivity': reflectivity_signature,
                    'median_reflectivity': median_reflectivity
                })
        
        return pedestrians

def extract_raw_predictions(config_path, weights_path, output_path):
    logger = create_logger()
    
    # 1. Setup Dataset
    orig_cwd = os.getcwd()
    os.chdir(os.path.join(orig_cwd, 'PFSD/tools'))
    config_abs_path = os.path.join(orig_cwd, config_path)
    cfg_from_yaml_file(config_abs_path, cfg)
    os.chdir(orig_cwd)
    
    # Let the config file dictate the DATA_PATH.
    cfg.DATA_CONFIG.DATA_PATH = os.path.abspath(os.path.join(orig_cwd, cfg.DATA_CONFIG.DATA_PATH))
    # We add v1.0-mini only if it's not already there
    if not cfg.DATA_CONFIG.DATA_PATH.endswith('v1.0-mini'):
        cfg.DATA_CONFIG.DATA_PATH = os.path.join(cfg.DATA_CONFIG.DATA_PATH, 'v1.0-mini')
    
    actual_data_path = cfg.DATA_CONFIG.DATA_PATH
    
    cfg.DATA_CONFIG.VERSION = '' 
    # Don't hardcode INFO_PATH so we can support Waymo cross-dataset eval
    print(f"ACTUAL DATA PATH IS: {actual_data_path}")
    if 'waymo' in str(actual_data_path):
        cfg.DATA_CONFIG.INFO_PATH['test'] = ['waymo_infos_val.pkl']
    else:
        cfg.DATA_CONFIG.INFO_PATH['test'] = ['pfsd_infos_10sweeps_val.pkl']
    dataset = NuScenesDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=Path(actual_data_path),
        logger=logger
    )

    # 2. Setup DL Model
    dl_tracker = PFSDTrackerWrapper(config_path, weights_path)
    
    # 3. Process all validation samples
    all_det_annos = []
    
    print(f"Extracting {len(dataset)} samples...")
    for i in range(len(dataset)):
        info = dataset.infos[i]
        token = info['token']
        
        data_dict = dataset[i]
        pc = data_dict['points']
        
        detections = dl_tracker.run_detection(pc)
        
        boxes_lidar = np.array([d['raw_box'] for d in detections]) if detections else np.zeros((0, 7))
        scores = np.array([d['score'] for d in detections]) if detections else np.zeros((0,))
        labels = np.array([9] * len(detections)) if detections else np.zeros((0,))
        reflectivities = np.array([d.get('reflectivity', 0.0) for d in detections]) if detections else np.zeros((0,))
        median_reflectivities = np.array([d.get('median_reflectivity', 0.0) for d in detections]) if detections else np.zeros((0,))
        
        all_det_annos.append({
            'boxes_lidar': boxes_lidar,
            'score': scores,
            'pred_labels': labels,
            'reflectivity': reflectivities,
            'median_reflectivity': median_reflectivities,
            'metadata': {'token': token}
        })
        
        if i % 50 == 0:
            print(f"Frame {i}/{len(dataset)}...")

    # 4. Save Raw Detections
    import pickle
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(all_det_annos, f)
    
    print(f"Raw detections saved to {output_path}")

if __name__ == "__main__":
    import sys
    cfg_p = sys.argv[1]
    w_p = sys.argv[2]
    out_p = sys.argv[3]
    
    extract_raw_predictions(cfg_p, w_p, out_p)
