import pickle
import os
import json
import numpy as np
import argparse


from core.person_tracker_appearance import PersonTracker
from tools.pfsd_evaluator import PFSDEvaluator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_pkl', type=str, default='results/raw_detections_cp_baseline_with_median.pkl')
    parser.add_argument('--dist_thresh', type=float, default=30.0)
    parser.add_argument('--far_score', type=float, default=0.6)
    parser.add_argument('--near_score', type=float, default=0.4)
    parser.add_argument('--refl_weight', type=float, default=1.5)
    parser.add_argument('--output_dir', type=str, default='results/ablation_test')
    args = parser.parse_args()

    print(f"Loading raw detections from {args.raw_pkl}...")
    with open(args.raw_pkl, 'rb') as f:
        raw_annos = pickle.load(f)

    # Apply Adaptive Thresholding (Frontier 3)
    filtered_annos = []
    for frame in raw_annos:
        boxes = frame['boxes_lidar']
        scores = frame['score']
        labels = frame['pred_labels']
        refls = frame['reflectivity']
        median_refls = frame.get('median_reflectivity', [])
        
        keep_idx = []
        for i in range(len(boxes)):
            center = boxes[i][:3]
            dist = np.sqrt(center[0]**2 + center[1]**2)
            thresh = args.far_score if dist > args.dist_thresh else args.near_score
            if scores[i] > thresh:
                keep_idx.append(i)
        
        filtered_annos.append({
            'boxes_lidar': boxes[keep_idx],
            'score': scores[keep_idx],
            'pred_labels': labels[keep_idx],
            'reflectivity': refls[keep_idx] if len(refls) > 0 else refls,
            'median_reflectivity': median_refls[keep_idx] if len(median_refls) > 0 else median_refls,
            'metadata': frame['metadata']
        })

    # Initialize tracker with Reflectivity Penalty (Frontier 1)
    tracker = PersonTracker(refl_penalty_weight=args.refl_weight)
    
    
    # Auto-detect dataroot (Triton vs Local)
    triton_path = '/scratch/work/pinomae1/data/pfsd/v1.0-mini'
    local_path = '/home/eelis/Desktop/Diplomityö/peopleTrackingWithLidar/PFSD/data/nuscenes/v1.0-mini'
    nusc_root = triton_path if os.path.exists(triton_path) else local_path
    
    evaluator = PFSDEvaluator(nusc_root=nusc_root, version='v1.0-mini')

    print(f"Running multi-modal tracking with refl_weight={args.refl_weight}...")
    nusc_results = evaluator.convert_to_nusc_tracking_format(filtered_annos, tracker)

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, 'tracking_results.json')
    submission = {
        'meta': {'use_camera': False, 'use_lidar': True, 'use_radar': False, 'use_map': False, 'use_external_tracking': False},
        'results': nusc_results
    }
    with open(results_path, 'w') as f:
        json.dump(submission, f)
    
    print("Evaluating metrics...")
    metrics = evaluator.run_evaluation(results_path, args.output_dir)
    print(f"\n--- Ablation Results ---")
    print(f"Params: Dist>{args.dist_thresh}m -> Score {args.far_score} | Near Score {args.near_score} | Refl Weight {args.refl_weight}")
    print(f"AMOTA: {metrics['amota']:.3f} | AMOTP: {metrics['amotp']:.3f} | RECALL: {metrics['recall']:.3f}")

if __name__ == "__main__":
    main()
