import json
import numpy as np
from collections import defaultdict
import sys
import os

def compute_hota(preds_path, gt_path, dist_thresh=2.0):
    with open(preds_path, 'r') as f:
        preds_data = json.load(f)['results']
        
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)['results']
        
    global_tp = 0
    global_fp = 0
    global_fn = 0
    loca_sum = 0.0
    
    gt_trajectories = defaultdict(list)
    pred_trajectories = defaultdict(list)
    matches = defaultdict(list)
    
    all_tokens = set(preds_data.keys()).union(set(gt_data.keys()))
    
    for token in all_tokens:
        preds = preds_data.get(token, [])
        gts = gt_data.get(token, [])
        
        preds = [p for p in preds if p['tracking_name'] == 'pedestrian']
        gts = [g for g in gts if g['tracking_name'] == 'pedestrian']
        
        for p in preds: pred_trajectories[p['tracking_id']].append(token)
        for g in gts: gt_trajectories[g['tracking_id']].append(token)
        
        if len(preds) == 0 or len(gts) == 0:
            global_fp += len(preds)
            global_fn += len(gts)
            continue
            
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
                matches[token].append((gts[i]['tracking_id'], preds[j]['tracking_id']))
                # LocA: Similarity score based on center distance
                sim = max(0.0, 1.0 - (cost / dist_thresh))
                loca_sum += sim
                
        global_tp += len(matched_g)
        global_fp += len(preds) - len(matched_p)
        global_fn += len(gts) - len(matched_g)
        
    det_a = global_tp / max(1, (global_tp + global_fp + global_fn))
    
    tpa_counts = defaultdict(int)
    for token, m_list in matches.items():
        for gt_id, pred_id in m_list:
            tpa_counts[(gt_id, pred_id)] += 1
            
    ass_a_sum = 0.0
    valid_c_count = 0
    
    for (gt_id, pred_id), tpa in tpa_counts.items():
        fpa = len(pred_trajectories[pred_id]) - tpa
        fna = len(gt_trajectories[gt_id]) - tpa
        
        a = tpa / max(1, (tpa + fpa + fna))
        
        ass_a_sum += a * tpa
        valid_c_count += tpa
        
    ass_a = ass_a_sum / max(1, valid_c_count)
    hota = np.sqrt(det_a * ass_a)
    loca = loca_sum / max(1, global_tp)
    
    print("=== HOTA Evaluation ===")
    print(f"HOTA: {hota:.4f}")
    print(f"DetA: {det_a:.4f}")
    print(f"AssA: {ass_a:.4f}")
    print(f"LocA: {loca:.4f}")
    print(f"Global TP:  {global_tp}")
    print(f"Global FP:  {global_fp}")
    print(f"Global FN:  {global_fn}")
    
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python compute_pfsd_hota.py <preds.json> <gt.json>")
        sys.exit(1)
    compute_hota(sys.argv[1], sys.argv[2])
