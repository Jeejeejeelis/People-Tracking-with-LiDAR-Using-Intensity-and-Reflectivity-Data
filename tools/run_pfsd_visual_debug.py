import numpy as np
import cv2
import os
from pathlib import Path
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from core.person_tracker import PersonTracker
from core.dl_tracker import OpenPCDetTracker
from run_pfsd_centerpoint import CenterPointPFSDTracker

class PFSDProjection:
    def __init__(self, nusc):
        self.nusc = nusc

    def project_lidar_to_image(self, points, cam_token, lidar_token):
        # 1. Lidar to Ego
        lidar_data = self.nusc.get('sample_data', lidar_token)
        l_calib = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        l_R = Quaternion(l_calib['rotation']).rotation_matrix
        l_T = np.array(l_calib['translation'])
        
        points_ego = (np.dot(l_R, points[:, :3].T).T + l_T)
        
        # 2. Ego to Cam
        cam_data = self.nusc.get('sample_data', cam_token)
        c_calib = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        c_R = Quaternion(c_calib['rotation']).rotation_matrix
        c_T = np.array(c_calib['translation'])
        
        points_cam = np.dot(c_R.T, (points_ego - c_T).T).T
        
        # 3. Project to image
        K = np.array(c_calib['camera_intrinsic'])
        
        mask = points_cam[:, 2] > 0.1
        points_cam_f = points_cam[mask]
        
        points_img = np.dot(K, points_cam_f.T).T
        points_img[:, 0] /= points_img[:, 2]
        points_img[:, 1] /= points_img[:, 2]
        
        return points_img[:, :2], points_cam_f[:, 2], mask

def run_pfsd_visual_debug(scene_name="10", num_frames=None, config_path=None, weights_path=None, out_dir=None):
    nusc = NuScenes(version='v1.0-mini', dataroot='OpenPCDet/data/nuscenes/v1.0-mini', verbose=False)
    proj = PFSDProjection(nusc)
    
    os.makedirs(out_dir, exist_ok=True)
    tracker = CenterPointPFSDTracker(config_path, weights_path)
    
    from core.person_tracker import PersonTracker
    person_tracker = PersonTracker(refl_penalty_weight=0.1, distance_threshold=30.0)
    
    from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
    from pcdet.config import cfg
    dataset_cfg = cfg.DATA_CONFIG
    dataset_cfg.DATA_PATH = 'OpenPCDet/data/nuscenes/v1.0-mini'
    dataset_cfg.VERSION = ''
    dataset_cfg.INFO_PATH['test'] = ['pfsd_infos_10sweeps_val.pkl']
    dataset_cfg.MAX_SWEEPS = 10

    all_classes = ['car','truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    
    import logging
    logger = logging.getLogger('visual_debug')
    dataset = NuScenesDataset(
        dataset_cfg=dataset_cfg,
        class_names=all_classes,
        training=False,
        root_path=Path('OpenPCDet/data/nuscenes/v1.0-mini'),
        logger=logger
    )
    print(f"DEBUG: Dataset mode: {dataset.mode}")
    print(f"DEBUG: Number of infos loaded: {len(dataset.infos)}")
    if len(dataset.infos) > 0:
        print(f"DEBUG: First info token: {dataset.infos[0]['token']}")

    # Token to scene mapping from TSV
    token_to_scene = {}
    if os.path.exists('core/token_to_scene.tsv'):
        with open('core/token_to_scene.tsv', 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    token_to_scene[parts[0]] = parts[1]

    # Map sample tokens to dataset indices
    token_to_idx = {info['token']: i for i, info in enumerate(dataset.infos)}
    
    # Get all scenes that actually have data in dataset.infos
    valid_scene_tokens = sorted(list(set(token_to_scene[t] for t in token_to_idx.keys() if t in token_to_scene)))
    
    if not valid_scene_tokens:
        print("Error: No valid scenes found in dataset.infos!")
        return

    # Find scene by index or name
    try:
        scene_id_int = int(scene_name)
        if scene_id_int < len(valid_scene_tokens):
            target_scene_token = valid_scene_tokens[scene_id_int]
        else:
            target_scene_token = valid_scene_tokens[0]
    except ValueError:
        if scene_name in valid_scene_tokens:
            target_scene_token = scene_name
        else:
            target_scene_token = valid_scene_tokens[0]

    print(f"Visualizing Scene: {target_scene_token}")

    # Get all samples for this scene that are in dataset.infos
    scene_samples = [t for t, s in token_to_scene.items() if s == target_scene_token and t in token_to_idx]
    # Sort by timestamp
    scene_samples.sort(key=lambda t: nusc.get('sample', t)['timestamp'])

    frame_id = 0
    for sample_token in scene_samples:
        if num_frames is not None and frame_id >= num_frames:
            break
            
        sample = nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        cam_token = sample['data']['CAM_FRONT_LEFT']
        
        # Load image
        cam_data = nusc.get('sample_data', cam_token)
        img_path = os.path.join(nusc.dataroot, cam_data['filename'])
        img = cv2.imread(img_path)
        
        # Get points from dataset (handles multi-sweeps)
        data_dict = dataset[token_to_idx[sample_token]]
        pc = data_dict['points']
        
        # DL tracker handles detection internally
        detections = tracker.run_detection(pc)
        
        # Add dummy reflectivity to detections to prevent KeyError, though visualizer doesn't strictly need it to look good
        for d in detections:
            d['reflectivity'] = 0.0
            
        tracked_peds = person_tracker.track(detections)
        print(f"Frame {frame_id}: {len(detections)} detections, {len(tracked_peds)} tracked.")
        
        # Draw tracked pedestrians
        def get_color_for_id(track_id):
            import colorsys
            # Golden ratio conjugate for well-distributed hues
            hue = (track_id * 0.618033988749895) % 1.0
            s = 0.85
            v = 0.9
            r, g, b = colorsys.hsv_to_rgb(hue, s, v)
            return (int(b * 255), int(g * 255), int(r * 255)) # BGR for OpenCV

        for ped in tracked_peds:
            p_min, p_max = ped['bbox']
            color = get_color_for_id(ped['id'])
            score = ped.get('score', 1.0)
            label = f"P{ped['id']} ({score:.2f})"
            
            # Create 8 corners of the 3D bbox
            corners = np.array([
                [p_min[0], p_min[1], p_min[2]], [p_max[0], p_min[1], p_min[2]],
                [p_max[0], p_max[1], p_min[2]], [p_min[0], p_max[1], p_min[2]],
                [p_min[0], p_min[1], p_max[2]], [p_max[0], p_min[1], p_max[2]],
                [p_max[0], p_max[1], p_max[2]], [p_min[0], p_max[1], p_max[2]]
            ])
            
            pts_2d, _, mask = proj.project_lidar_to_image(corners, cam_token, lidar_token)
            
            # Debug first detection
            if ped['id'] == tracked_peds[0]['id']:
                print(f"DEBUG: Ped 0 Center: {ped['center']}")
                print(f"DEBUG: Ped 0 2D Points (front only): {pts_2d}")
            
            # Draw if we have at least 1 point in front
            if len(pts_2d) == 0: continue
            
            pts_2d_all = np.zeros((8, 2), dtype=int)
            pts_2d_all[mask] = pts_2d.astype(int)
            
            if not all(mask):
                valid_pt = pts_2d[0].astype(int)
                pts_2d_all[~mask] = valid_pt

            # Draw 3D Bounding Box in 2D
            thickness = 3
            for i in range(4):
                # Bottom square
                cv2.line(img, tuple(pts_2d_all[i]), tuple(pts_2d_all[(i+1)%4]), color, thickness)
                # Top square
                cv2.line(img, tuple(pts_2d_all[i+4]), tuple(pts_2d_all[(i+1)%4+4]), color, thickness)
                # Vertical lines
                cv2.line(img, tuple(pts_2d_all[i]), tuple(pts_2d_all[i+4]), color, thickness)
            
            # Label
            center_3d = ped['center'].reshape(1, 3)
            center_2d, _, _ = proj.project_lidar_to_image(center_3d, cam_token, lidar_token)
            if len(center_2d) > 0:
                cv2.putText(img, label, (int(center_2d[0][0]), int(center_2d[0][1])), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


        out_path = os.path.join(out_dir, f"{frame_id:03d}.jpg")
        cv2.imwrite(out_path, img)
        print(f"Saved {out_path}")
        
        sample_token = sample['next']
        frame_id += 1

    print(f"\nPFSD Visual Debug for Scene {scene_name}  complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="10")
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    run_pfsd_visual_debug(
        scene_name=args.scene,
        num_frames=args.frames,
        config_path=args.config,
        weights_path=args.weights,
        out_dir=args.out_dir
    )
