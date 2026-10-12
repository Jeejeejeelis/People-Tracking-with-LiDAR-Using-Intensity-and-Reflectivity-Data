import pickle
import numpy as np
from scipy.spatial.distance import cdist

with open('data/waymo_processed/v1.0-mini/waymo_infos_val.pkl', 'rb') as f:
    gt_infos = pickle.load(f)
    
with open('results/waymo_predictions.pkl', 'rb') as f:
    preds = pickle.load(f)

total_gt = 0
total_preds = 0
true_positives = 0

for gt_frame, pred_frame in zip(gt_infos, preds):
    gt_boxes = gt_frame['gt_boxes']
    pred_boxes = pred_frame['boxes_lidar']
    
    total_gt += len(gt_boxes)
    total_preds += len(pred_boxes)
    
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        continue
        
    # Compute center distances (BEV X, Y)
    gt_centers = gt_boxes[:, :2]
    pred_centers = pred_boxes[:, :2]
    
    dists = cdist(gt_centers, pred_centers)
    
    # Simple greedy match: distance < 1.0 meter
    matched_gt = set()
    matched_pred = set()
    
    # Sort distances
    for r in range(dists.shape[0]):
        for c in range(dists.shape[1]):
            if dists[r, c] < 1.0 and r not in matched_gt and c not in matched_pred:
                matched_gt.add(r)
                matched_pred.add(c)
                true_positives += 1

print(f"Total GT: {total_gt}")
print(f"Total Preds: {total_preds}")
print(f"True Positives: {true_positives}")
print(f"False Positives: {total_preds - true_positives}")
print(f"False Negatives: {total_gt - true_positives}")
print(f"Actual Recall: {(true_positives / total_gt) * 100:.2f}%")
print(f"Precision: {(true_positives / total_preds) * 100:.2f}%" if total_preds > 0 else "Precision: 0.0%")
