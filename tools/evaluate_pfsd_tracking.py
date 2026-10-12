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
            if pred_labels[i] == self.ped_label:
                box = pred_boxes[i]
                center = box[:3]
                dist = np.sqrt(center[0]**2 + center[1]**2)
                
                # Frontier 3: Adaptive Range-Dependent Confidence Thresholding
                # If an object is far away (>30m), require higher confidence to combat d^2 ghosting
                adaptive_thresh = 0.6 if dist > 30.0 else 0.4
                
                if pred_scores[i] > adaptive_thresh:
                    dims = box[3:6]
                    p_min = center - (dims / 2)
                    p_max = center + (dims / 2)
                    
                    # Frontier 1: Reflectivity Signature Extraction
                    # Find raw LiDAR points inside this bounding box
                    mask = (points[:, 0] >= p_min[0]) & (points[:, 0] <= p_max[0]) & \
                           (points[:, 1] >= p_min[1]) & (points[:, 1] <= p_max[1]) & \
                           (points[:, 2] >= p_min[2]) & (points[:, 2] <= p_max[2])
                    
                    points_in_box = points[mask]
                    if len(points_in_box) > 0:
                        reflectivity_signature = float(np.mean(points_in_box[:, 3]))
                    else:
                        reflectivity_signature = 0.0
                    
                    pedestrians.append({
                        'center': center,
                        'bbox': (p_min, p_max),
                        'score': pred_scores[i],
                        'raw_box': box,
                        'reflectivity': reflectivity_signature
                    })
        
        return pedestrians

def evaluate_pfsd_tracking(config_path, weights_path, output_dir, thresh=None):
    logger = create_logger()
    
    # 1. Setup Dataset
    cfg_from_yaml_file(config_path, cfg)
    if thresh is not None:
        cfg.MODEL.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH = thresh
    # Ensure correct data paths for local run or Triton
    root_dir = Path(__file__).resolve().parent.parent
    
    if not Path(cfg.DATA_CONFIG.DATA_PATH).exists():
        if (root_dir / 'PFSD/data/nuscenes/v1.0-mini').exists():
            cfg.DATA_CONFIG.DATA_PATH = str(root_dir / 'PFSD/data/nuscenes/v1.0-mini')
        elif (root_dir / 'OpenPCDet/data/nuscenes/v1.0-mini').exists():
            cfg.DATA_CONFIG.DATA_PATH = str(root_dir / 'OpenPCDet/data/nuscenes/v1.0-mini')
            
    if not cfg.DATA_CONFIG.DATA_PATH.endswith('v1.0-mini'):
        cfg.DATA_CONFIG.DATA_PATH = os.path.join(cfg.DATA_CONFIG.DATA_PATH, 'v1.0-mini')
        
    actual_data_path = cfg.DATA_CONFIG.DATA_PATH
    
    cfg.DATA_CONFIG.VERSION = '' 
    cfg.DATA_CONFIG.INFO_PATH['test'] = ['pfsd_infos_10sweeps_val.pkl']
    
    dataset = NuScenesDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=Path(actual_data_path),
        logger=logger
    )

    # 2. Setup Tracker
    dl_tracker = PFSDTrackerWrapper(config_path, weights_path)
    person_tracker = PersonTracker()
    
    # 3. Process all validation samples
    all_det_annos = []
    
    print(f"Processing {len(dataset)} samples...")
    for i in range(len(dataset)):
        info = dataset.infos[i]
        token = info['token']
        
        data_dict = dataset[i]
        pc = data_dict['points']
        
        detections = dl_tracker.run_detection(pc)
        
        # We need to save detections in a format that PFSDEvaluator can use
        # PFSDEvaluator.convert_to_nusc_tracking_format expects:
        # det_annos: list of dicts with 'boxes_lidar', 'score', 'pred_labels', 'metadata'
        
        boxes_lidar = np.array([d['raw_box'] for d in detections]) if detections else np.zeros((0, 7))
        scores = np.array([d['score'] for d in detections]) if detections else np.zeros((0,))
        labels = np.array([9] * len(detections)) if detections else np.zeros((0,))
        reflectivities = np.array([d.get('reflectivity', 0.0) for d in detections]) if detections else np.zeros((0,))
        
        all_det_annos.append({
            'boxes_lidar': boxes_lidar,
            'score': scores,
            'pred_labels': labels,
            'reflectivity': reflectivities,
            'metadata': {'token': token}
        })
        
        if i % 50 == 0:
            print(f"Frame {i}/{len(dataset)}...")

    # 4. Convert and Evaluate
    evaluator = PFSDEvaluator(nusc_root=actual_data_path, version='v1.0-mini')
    
    # Group by scene as PFSDEvaluator does it internaly? 
    # Actually PFSDEvaluator.convert_to_nusc_tracking_format takes the list and a tracker.
    
    print("Running tracker and converting to nuScenes format...")
    nusc_results = evaluator.convert_to_nusc_tracking_format(all_det_annos, person_tracker)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'tracking_results.json')
    submission = {
        'meta': {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external_tracking': False
        },
        'results': nusc_results
    }
    with open(results_path, 'w') as f:
        json.dump(submission, f)
    
    print(f"Results saved to {results_path}")
    
    # 5. Run Evaluation
    print("Calculating metrics...")
    metrics = evaluator.run_evaluation(results_path, output_dir)
    
    print("\n--- Evaluation Summary ---")
    print(metrics)

if __name__ == "__main__":
    import sys
    cfg_p = sys.argv[1] if len(sys.argv) > 1 else "OpenPCDet/tools/cfgs/nuscenes_models/pfsd_pp_multihead.yaml"
    w_p = sys.argv[2] if len(sys.argv) > 2 else "weights/pp_nuscenes.pth"
    out_d = sys.argv[3] if len(sys.argv) > 3 else "results/eval_pfsd_pp"
    thresh = None
    if len(sys.argv) > 4:
        thresh = float(sys.argv[4])
    
    evaluate_pfsd_tracking(cfg_p, w_p, out_d, thresh=thresh)
