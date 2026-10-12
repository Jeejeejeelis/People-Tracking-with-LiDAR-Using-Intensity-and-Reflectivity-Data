import json
import numpy as np
import pickle
import os
from nuscenes.nuscenes import NuScenes

def export_gt():
    # Detect if we are on Triton
    if os.path.exists('/scratch/work/pinomae1/data/pfsd/v1.0-mini'):
        dataroot = '/scratch/work/pinomae1/data/pfsd/v1.0-mini'
    else:
        dataroot = 'PFSD/data/nuscenes/v1.0-mini'
        
    print(f"Using dataroot: {dataroot}")
    
    # Triton symlink structure sometimes causes Nuscenes to look for v1.0-mini/v1.0-mini
    try:
        nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=True)
    except AssertionError:
        # Fallback for Triton if v1.0-mini is nested
        nusc = NuScenes(version='v1.0-mini', dataroot=os.path.join(dataroot, 'v1.0-mini'), verbose=True)
    
    info_path = os.path.join(dataroot, 'pfsd_infos_10sweeps_val.pkl')
    with open(info_path, 'rb') as f:
        infos = pickle.load(f)
        
    results = {}
    
    for info in infos:
        curr_token = info['token']
        sample = nusc.get('sample', curr_token)
        results[curr_token] = []
        
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            cat = ann['category_name']
            
            if 'pedestrian' not in cat:
                continue
                
            vel = nusc.box_velocity(ann_token)[:2].tolist()
            if np.isnan(vel[0]): vel = [0.0, 0.0]
                
            res = {
                'sample_token': curr_token,
                'translation': ann['translation'],
                'size': ann['size'],
                'rotation': ann['rotation'],
                'velocity': vel,
                'tracking_id': ann['instance_token'],
                'tracking_name': 'pedestrian',
                'tracking_score': 1.0
            }
            
            results[curr_token].append(res)
            
    out_dict = {
        'meta': {'use_camera': False, 'use_lidar': True, 'use_radar': False, 'use_map': False, 'use_external_tracking': False},
        'results': results
    }
    
    out_path = os.path.join(dataroot, 'pfsd_tracking_gt.json')
    with open(out_path, 'w') as f:
        json.dump(out_dict, f, indent=2)
        
    print(f"Exported GT with {len(results)} frames to {out_path}.")

if __name__ == '__main__':
    export_gt()
