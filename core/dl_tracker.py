import numpy as np
import cv2
import torch
import os
from scipy.optimize import linear_sum_assignment

# Note: These imports will work once OpenPCDet is installed on the GPU machine.
try:
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import common_utils
    from pcdet.datasets import DatasetTemplate
    HAS_PCDET = True
except ImportError:
    print("Warning: OpenPCDet (pcdet) is not installed. You are in 'prepare mode'.")
    HAS_PCDET = False

class DummyDataset(DatasetTemplate):
    """
    A minimal dataset wrapper to feed a single point cloud directly into OpenPCDet 
    without needing the full KITTI info files during inference testing.
    """
    def __init__(self, dataset_cfg, class_names, training=False, root_path=None, ext='.bin'):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=None)
        self.ext = ext

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {}


class OpenPCDetTracker:
    def __init__(self, config_path, weights_path, intensity_mode='none'):
        """
        Initializes the OpenPCDet model (PointPillars or CenterPoint) and the Hungarian Tracker.
        intensity_mode: 'none', 'raw', or 'compensated'
        """
        # Tracking state (Reusing the logic from our PersonTracker)
        self.next_track_id = 1
        self.tracks = [] 
        self.max_missed = 3
        self.dist_thresh = 2.0 # Slightly higher threshold for DL models since they capture full extent better
        
        self.intensity_mode = intensity_mode
        self.model = None

        if HAS_PCDET:
            self._init_model(config_path, weights_path)

    def _init_model(self, config_path, weights_path):
        import os
        orig_cwd = os.getcwd()
        # Navigate to PFSD/tools so OpenPCDet config parser can resolve relative paths
        if os.path.exists('PFSD/tools'):
            os.chdir('PFSD/tools')
            config_path_abs = os.path.join(orig_cwd, config_path)
        else:
            config_path_abs = config_path

        cfg_from_yaml_file(config_path_abs, cfg)
        
        if os.path.exists(orig_cwd):
            os.chdir(orig_cwd)

        print(f"Loading OpenPCDet model from {weights_path}...")
        logger = common_utils.create_logger()
        
        # Build dummy dataset to satisfy the model builder
        dataset = DummyDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False)
        
        # Build the network and load weights
        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
        self.model.load_params_from_file(filename=weights_path, logger=logger, to_cpu=False)
        self.model.cuda()
        self.model.eval()
        print("Model loaded successfully and moved to GPU.")

    def get_compensated_reflectivity(self, lidar_data):
        """
        lidar_data: (N, 4) array [x, y, z, intensity]
        returns: (N,) array of distance-compensated reflectivity
        """
        x, y, z = lidar_data[:, 0], lidar_data[:, 1], lidar_data[:, 2]
        intensity = lidar_data[:, 3]
        distance = np.sqrt(x**2 + y**2 + z**2)
        reflectivity = intensity * (distance**2)
        denom = (np.max(reflectivity) - np.min(reflectivity))
        if denom > 0:
            reflectivity = (reflectivity - np.min(reflectivity)) / denom
        return reflectivity

    def run_detection(self, points):
        """
        Runs the OpenPCDet model on a single point cloud frame.
        """
        if not HAS_PCDET:
            # Return dummy detections if we're on the CPU laptop
            return []

        # The pre-trained PointPillars model expects 4 features (x, y, z, intensity).
        if points.shape[1] < 4:
            fill = np.zeros((points.shape[0], 4 - points.shape[1]), dtype=points.dtype)
            points = np.hstack((points, fill))
        
        if self.intensity_mode == 'none':
            # Phase 1: Spatial Only - Zero out the intensity feature
            points[:, 3] = 0
        elif self.intensity_mode == 'raw':
            # Use raw intensity as is
            pass
        elif self.intensity_mode == 'compensated':
            # Use distance-compensated reflectivity as the 4th feature
            points[:, 3] = self.get_compensated_reflectivity(points)

        with torch.no_grad():
            # Use the dataset's prepare_data to handle voxelization and other processing
            input_dict = {
                'points': points,
                'frame_id': np.array(['000000']),
            }
            
            data_dict = self.model.dataset.prepare_data(data_dict=input_dict)
            data_dict = self.model.dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            
            # Forward pass
            pred_dicts, _ = self.model.forward(data_dict)
            
            # Parse output
            pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy() # [x, y, z, dx, dy, dz, heading]
            pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()
            pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()

        pedestrians = []
        for i in range(len(pred_boxes)):
            # OpenPCDet classes usually: 1: Car, 2: Pedestrian, 3: Cyclist
            if pred_labels[i] == 2 and pred_scores[i] > 0.4: # Filter by class and confidence
                box = pred_boxes[i]
                center = box[:3] # (x, y, z)
                dims = box[3:6]  # (l, w, h)
                
                # Approximate p_min and p_max for compatibility with our visualization script
                p_min = center - (dims / 2)
                p_max = center + (dims / 2)
                
                pedestrians.append({
                    'center': center,
                    'bbox': (p_min, p_max),
                    'points': np.array([center]), # OpenPCDet doesn't output the raw cluster points, just the box.
                    'score': pred_scores[i]
                })

        return pedestrians

    def track(self, current_pedestrians):
        """
        Data association (Nearest Neighbor using Hungarian algorithm) for tracking.
        Reusing the same robust logic from our DBSCAN tracker.
        """
        if not self.tracks:
            for ped in current_pedestrians:
                self.tracks.append({
                    'id': self.next_track_id,
                    'center': ped['center'],
                    'bbox': ped['bbox'],
                    'missed_frames': 0
                })
                self.next_track_id += 1
            return self.tracks

        if not current_pedestrians:
            for t in self.tracks:
                t['missed_frames'] += 1
            self.tracks = [t for t in self.tracks if t['missed_frames'] <= self.max_missed]
            return self.tracks

        cost_matrix = np.zeros((len(self.tracks), len(current_pedestrians)))
        for i, t in enumerate(self.tracks):
            for j, d in enumerate(current_pedestrians):
                cost_matrix[i, j] = np.linalg.norm(t['center'] - d['center'])

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_detections = set(range(len(current_pedestrians)))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < self.dist_thresh:
                self.tracks[r]['center'] = current_pedestrians[c]['center']
                self.tracks[r]['bbox'] = current_pedestrians[c]['bbox']
                self.tracks[r]['missed_frames'] = 0
                unmatched_tracks.remove(r)
                unmatched_detections.remove(c)

        for r in unmatched_tracks:
            self.tracks[r]['missed_frames'] += 1

        for c in unmatched_detections:
            ped = current_pedestrians[c]
            self.tracks.append({
                'id': self.next_track_id,
                'center': ped['center'],
                'bbox': ped['bbox'],
                'missed_frames': 0
            })
            self.next_track_id += 1

        self.tracks = [t for t in self.tracks if t['missed_frames'] <= self.max_missed]
        # Return all tracks that are not dead (allowing interpolation)
        return self.tracks

if __name__ == "__main__":
    print("OpenPCDet DL Tracker script ready. Awaiting GPU execution.")
