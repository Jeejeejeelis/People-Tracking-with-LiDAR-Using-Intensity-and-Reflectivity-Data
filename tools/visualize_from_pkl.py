import numpy as np
import cv2
import os
import pickle
from pathlib import Path
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from core.person_tracker import PersonTracker

class PFSDProjection:
    def __init__(self, nusc):
        self.nusc = nusc

    def project_lidar_to_image(self, points, cam_token, lidar_token):
        lidar_data = self.nusc.get('sample_data', lidar_token)
        l_calib = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        l_R = Quaternion(l_calib['rotation']).rotation_matrix
        l_T = np.array(l_calib['translation'])
        
        points_ego = (np.dot(l_R, points[:, :3].T).T + l_T)
        
        cam_data = self.nusc.get('sample_data', cam_token)
        c_calib = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        c_R = Quaternion(c_calib['rotation']).rotation_matrix
        c_T = np.array(c_calib['translation'])
        
        points_cam = np.dot(c_R.T, (points_ego - c_T).T).T
        
        K = np.array(c_calib['camera_intrinsic'])
        
        mask = points_cam[:, 2] > 0.1
        points_cam_f = points_cam[mask]
        
        points_img = np.dot(K, points_cam_f.T).T
        points_img[:, 0] /= points_img[:, 2]
        points_img[:, 1] /= points_img[:, 2]
        
        return points_img[:, :2], points_cam_f[:, 2], mask

def get_color_for_id(track_id):
    import colorsys
    hue = (track_id * 0.618033988749895) % 1.0
    s = 0.85
    v = 0.9
    r, g, b = colorsys.hsv_to_rgb(hue, s, v)
    return (int(b * 255), int(g * 255), int(r * 255))

def run_pkl_visual(scene_name="10", pkl_path="", out_dir=""):
    nusc = NuScenes(version='v1.0-mini', dataroot='OpenPCDet/data/nuscenes/v1.0-mini', verbose=False)
    proj = PFSDProjection(nusc)
    
    os.makedirs(out_dir, exist_ok=True)
    
    with open(pkl_path, 'rb') as f:
        raw_annos = pickle.load(f)
        
    tracker = PersonTracker(refl_penalty_weight=0.1) # tuned parameter from the table (0.1x Refl)
    
    # Scene token mapping
    token_to_scene = {}
    if os.path.exists('core/token_to_scene.tsv'):
        with open('core/token_to_scene.tsv', 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    token_to_scene[parts[0]] = parts[1]

    with open('OpenPCDet/data/nuscenes/v1.0-mini/pfsd_infos_10sweeps_val.pkl', 'rb') as f:
        infos = pickle.load(f)
    token_to_idx = {info['token']: i for i, info in enumerate(infos)}
    valid_scene_tokens = sorted(list(set(token_to_scene[t] for t in token_to_idx.keys() if t in token_to_scene)))

    try:
        scene_id_int = int(scene_name)
        target_scene_token = valid_scene_tokens[scene_id_int]
    except:
        target_scene_token = valid_scene_tokens[0]

    scene_samples = [t for t, s in token_to_scene.items() if s == target_scene_token]
    scene_samples.sort(key=lambda t: nusc.get('sample', t)['timestamp'])
    
    # Filter PKL annos to match this scene
    # raw_annos contains all frames. We need to match by frame_id or token.
    # In extract_raw_predictions, they are just sequential. We assume they match the dataset order.
    # Wait, raw_annos usually has a 'frame_id' or 'metadata'.
    
    print(f"Visualizing Scene: {target_scene_token} from {pkl_path}")
    
    # Create a mapping from token to anno
    token_to_anno = {}
    for anno in raw_annos:
        if 'metadata' in anno and 'token' in anno['metadata']:
            token_to_anno[anno['metadata']['token']] = anno

    frame_id = 0
    for sample_token in scene_samples:
        if sample_token not in token_to_anno:
            print(f"Warning: sample_token {sample_token} not in pkl!")
            continue
            
        anno = token_to_anno[sample_token]
        
        sample = nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        cam_token = sample['data']['CAM_FRONT_LEFT']
        
        img_path = os.path.join(nusc.dataroot, nusc.get('sample_data', cam_token)['filename'])
        img = cv2.imread(img_path)
        
        # Build detections list
        detections = []
        if 'boxes_lidar' in anno and len(anno['boxes_lidar']) > 0:
            boxes = anno['boxes_lidar']
            scores = anno['score']
            labels = anno['pred_labels']
            for i in range(len(boxes)):
                if labels[i] == 9: # pedestrian
                    if scores[i] >= 0.4: # base threshold before tracking
                        x, y, z, dx, dy, dz, heading = boxes[i][:7]
                        detections.append({
                            'bbox': [[x-dx/2, y-dy/2, z-dz/2], [x+dx/2, y+dy/2, z+dz/2]],
                            'center': np.array([x, y, z]),
                            'score': float(scores[i])
                        })
        
        tracked_peds = tracker.track(detections)
        print(f"Frame {frame_id}: {len(detections)} detections, {len(tracked_peds)} tracked.")
        
        for ped in tracked_peds:
            p_min, p_max = ped['bbox']
            color = get_color_for_id(ped['id'])
            label = f"P{ped['id']} ({ped.get('score',1.0):.2f})"
            
            corners = np.array([
                [p_min[0], p_min[1], p_min[2]], [p_max[0], p_min[1], p_min[2]],
                [p_max[0], p_max[1], p_min[2]], [p_min[0], p_max[1], p_min[2]],
                [p_min[0], p_min[1], p_max[2]], [p_max[0], p_min[1], p_max[2]],
                [p_max[0], p_max[1], p_max[2]], [p_min[0], p_max[1], p_max[2]]
            ])
            pts_2d, _, mask = proj.project_lidar_to_image(corners, cam_token, lidar_token)
            if len(pts_2d) == 0: continue
            
            pts_2d_all = np.zeros((8, 2), dtype=int)
            pts_2d_all[mask] = pts_2d.astype(int)
            if not all(mask):
                valid_pt = pts_2d[0].astype(int)
                pts_2d_all[~mask] = valid_pt

            thickness = 3
            for i in range(4):
                cv2.line(img, tuple(pts_2d_all[i]), tuple(pts_2d_all[(i+1)%4]), color, thickness)
                cv2.line(img, tuple(pts_2d_all[i+4]), tuple(pts_2d_all[(i+1)%4+4]), color, thickness)
                cv2.line(img, tuple(pts_2d_all[i]), tuple(pts_2d_all[i+4]), color, thickness)
            
            center_3d = ped['center'].reshape(1, 3)
            center_2d, _, _ = proj.project_lidar_to_image(center_3d, cam_token, lidar_token)
            if len(center_2d) > 0:
                cv2.putText(img, label, (int(center_2d[0][0]), int(center_2d[0][1])), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out_path = os.path.join(out_dir, f"{frame_id:03d}.jpg")
        cv2.imwrite(out_path, img)
        frame_id += 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    run_pkl_visual(scene_name="10", pkl_path=args.pkl, out_dir=args.out_dir)
