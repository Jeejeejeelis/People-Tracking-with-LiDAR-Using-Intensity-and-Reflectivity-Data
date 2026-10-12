import json
import numpy as np
from collections import defaultdict
import sys
from scipy.optimize import linear_sum_assignment

def compute_idf1(preds_path, gt_path, dist_thresh=2.0):
    with open(preds_path, 'r') as f:
        preds_data = json.load(f)['results']
        
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)['results']
        
    total_pred_boxes = 0
    total_gt_boxes = 0
    
    # Track the number of frame-wise matches for each (gt_id, pred_id) pair
    tpa_counts = defaultdict(int)
    
    all_tokens = set(preds_data.keys()).union(set(gt_data.keys()))
    
    for token in all_tokens:
        preds = preds_data.get(token, [])
        gts = gt_data.get(token, [])
        
        preds = [p for p in preds if p['tracking_name'] == 'pedestrian']
        gts = [g for g in gts if g['tracking_name'] == 'pedestrian']
        
        total_pred_boxes += len(preds)
        total_gt_boxes += len(gts)
        
        if len(preds) == 0 or len(gts) == 0:
            continue
            
        # Match using center distance
        cost_matrix = np.zeros((len(gts), len(preds)))
        for i, g in enumerate(gts):
            for j, p in enumerate(preds):
                dist = np.linalg.norm(np.array(g['translation'][:2]) - np.array(p['translation'][:2]))
                cost_matrix[i, j] = dist
                
        matched_g = set()
        matched_p = set()
        
        flat_costs = []
        for i in range(len(gts)):
            for j in range(len(preds)):
                flat_costs.append((cost_matrix[i, j], i, j))
        flat_costs.sort(key=lambda x: x[0])
        
        for cost, i, j in flat_costs:
            if cost > dist_thresh:
                break
            if i not in matched_g and j not in matched_p:
                matched_g.add(i)
                matched_p.add(j)
                tpa_counts[(gts[i]['tracking_id'], preds[j]['tracking_id'])] += 1

    # Now we do the global bipartite matching to maximize IDTP
    # We need a list of unique gt_ids and pred_ids
    unique_gt_ids = list(set([k[0] for k in tpa_counts.keys()]))
    unique_pred_ids = list(set([k[1] for k in tpa_counts.keys()]))
    
    if len(unique_gt_ids) == 0 or len(unique_pred_ids) == 0:
        print("=== IDF1 Evaluation ===")
        print("IDF1: 0.0000")
        return
        
    gt_id_to_idx = {gid: i for i, gid in enumerate(unique_gt_ids)}
    pred_id_to_idx = {pid: i for i, pid in enumerate(unique_pred_ids)}
    
    # Cost matrix for global matching. We want to MAXIMIZE tpa, so we MINIMIZE negative tpa
    global_cost = np.zeros((len(unique_gt_ids), len(unique_pred_ids)))
    
    for (gt_id, pred_id), count in tpa_counts.items():
        r = gt_id_to_idx[gt_id]
        c = pred_id_to_idx[pred_id]
        global_cost[r, c] = -count  # Negative because linear_sum_assignment minimizes
        
    row_ind, col_ind = linear_sum_assignment(global_cost)
    
    idtp = 0
    for r, c in zip(row_ind, col_ind):
        # We only count it if they were matched at least once (cost < 0)
        if global_cost[r, c] < 0:
            idtp += int(-global_cost[r, c])
            
    idfp = total_pred_boxes - idtp
    idfn = total_gt_boxes - idtp
    
    idf1 = (2.0 * idtp) / max(1, (total_pred_boxes + total_gt_boxes))
    idp = idtp / max(1, total_pred_boxes)
    idr = idtp / max(1, total_gt_boxes)
    
    print("=== IDF1 Evaluation ===")
    print(f"IDF1: {idf1:.4f}")
    print(f"IDP:  {idp:.4f} (ID Precision)")
    print(f"IDR:  {idr:.4f} (ID Recall)")
    print(f"IDTP: {idtp}")
    print(f"IDFP: {idfp}")
    print(f"IDFN: {idfn}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python compute_pfsd_idf1.py <preds.json> <gt.json>")
        sys.exit(1)
    compute_idf1(sys.argv[1], sys.argv[2])
