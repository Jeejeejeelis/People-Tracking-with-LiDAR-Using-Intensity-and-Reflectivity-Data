import os
import subprocess
import re

models = [
    ('CenterPoint Baseline (5-feat: $I$)', 'results/raw_detections_cp_baseline.pkl'),
    ('CenterPoint Decoupled (6-feat: $I, d$)', 'results/raw_detections_cp_decoupled.pkl'),
    ('CenterPoint Enhanced (6-feat: $I, R$)', 'results/raw_detections_cp_enhanced.pkl'),
    ('CenterPoint Kitchen Sink (7-feat: $I, R, d$)', 'results/raw_detections_cp_kitchen_sink.pkl'),
    ('CenterPoint Swap (6-feat: $R, d$)', 'results/raw_detections_cp_swap.pkl'),
    
    ('VoxelNeXt Baseline (5-feat: $I$)', 'results/raw_detections_vn_baseline.pkl'),
    ('VoxelNeXt Decoupled (6-feat: $I, d$)', 'results/raw_detections_vn_decoupled.pkl'),
    ('VoxelNeXt Enhanced (6-feat: $I, R$)', 'results/raw_detections_vn_enhanced.pkl'),
    ('VoxelNeXt Kitchen Sink (7-feat: $I, R, d$)', 'results/raw_detections_vn_kitchen_sink.pkl'),
    ('VoxelNeXt Swap (6-feat: $R, d$)', 'results/raw_detections_vn_swap.pkl'),
    
    ('HMFN Baseline (5-feat: $I$)', 'results/raw_detections_hmfn_baseline.pkl'),
    ('HMFN Decoupled (6-feat: $I, d$)', 'results/raw_detections_hmfn_decoupled.pkl'),
    ('HMFN Enhanced (6-feat: $I, R$)', 'results/raw_detections_hmfn_enhanced.pkl'),
    ('HMFN Kitchen Sink (7-feat: $I, R, d$)', 'results/raw_detections_hmfn_kitchen_sink.pkl'),
    ('HMFN Swap (6-feat: $R, d$)', 'results/raw_detections_hmfn_swap.pkl'),

    ('PointPillars Baseline (5-feat: $I$)', 'results/raw_detections_pp_baseline.pkl'),
    ('PointPillars Swap (6-feat: $R, d$)', 'results/raw_detections_pp_swap.pkl')
]

GT_PATH = './OpenPCDet/data/nuscenes/v1.0-mini/pfsd_tracking_gt.json'

results_table = []

for name, pkl_path in models:
    if pkl_path is None or not os.path.exists(pkl_path):
        print(f"Skipping {name} (pkl not found or mode collapsed)")
        results_table.append((name, 'N/A', '0.0%', '0.00%', '0.00%', '0.00%', '0.00%', '0.00%'))
        continue

    print(f"\n=====================================")
    print(f"Optimizing {name}")
    print(f"=====================================")

    has_refl = "R" in name
    
    # Grid search options
    refl_weights = [0.0, 0.1, 0.3, 0.5] if has_refl else [0.0]
    dist_threshes = [100.0, 40.0, 30.0]
    
    best_amota = -1.0
    best_params = None
    best_json = ""
    
    for rw in refl_weights:
        for dt in dist_threshes:
            far_score = 0.6 if dt < 100.0 else 0.4
            out_dir = f"results/temp_optim_{rw}_{dt}"
            
            cmd = f"python tools/tune_tracking_frontiers.py --raw_pkl {pkl_path} --dist_thresh {dt} --far_score {far_score} --near_score 0.4 --refl_weight {rw} --output_dir {out_dir}"
            output = subprocess.check_output(cmd, shell=True, text=True)
            
            # Parse AMOTA
            amota = 0.0
            for line in output.split('\n'):
                if line.startswith("AMOTA:"):
                    amota = float(line.split()[1])
            
            print(f"  -> Dist: {dt}, Refl: {rw}x => AMOTA: {amota:.3f}")
            
            if amota > best_amota:
                best_amota = amota
                best_params = f"Dist: {dt if dt<100 else 'None'}, Refl: {rw}x"
                best_json = os.path.join(out_dir, 'tracking_results.json')
                
    # Now compute HOTA on best json
    hota_cmd = f"python tools/compute_pfsd_hota.py {best_json} {GT_PATH}"
    hota_out = subprocess.check_output(hota_cmd, shell=True, text=True)
    
    hota, deta, assa, loca = 0.0, 0.0, 0.0, 0.0
    for line in hota_out.split('\n'):
        if line.startswith("HOTA:"): hota = float(line.split()[1]) * 100
        if line.startswith("DetA:"): deta = float(line.split()[1]) * 100
        if line.startswith("AssA:"): assa = float(line.split()[1]) * 100
        if line.startswith("LocA:"): loca = float(line.split()[1]) * 100
        
    idf1_cmd = f"python tools/compute_pfsd_idf1.py {best_json} {GT_PATH}"
    idf1_out = subprocess.check_output(idf1_cmd, shell=True, text=True)
    idf1 = 0.0
    for line in idf1_out.split('\n'):
        if line.startswith("IDF1:"): idf1 = float(line.split()[1]) * 100
        
    print(f"WINNER for {name}: {best_params} (AMOTA: {best_amota*100:.1f}%)")
    results_table.append((name, best_params, f"{best_amota*100:.1f}%", f"{hota:.2f}%", f"{deta:.2f}%", f"{assa:.2f}%", f"{loca:.2f}%", f"{idf1:.2f}%"))

# Write Markdown output
md = "| Architecture | Input Features | Optimal Tracker Heuristic | AMOTA | HOTA | DetA | AssA | LocA | IDF1 |\n"
md += "|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|\n"
for row in results_table:
    name, params, amota, hota, deta, assa, loca, idf1 = row
    arch = name.split()[0]
    features = name.replace(arch + " ", "")
    if arch == "PointPillars": arch = "**PointPillars**"
    elif arch == "CenterPoint": arch = "**CenterPoint**"
    elif arch == "VoxelNeXt": arch = "**VoxelNeXt**"
    elif arch == "HMFN": arch = "**HMFN**"
    md += f"| {arch} | {features} | {params} | **{amota}** | {hota} | {deta} | {assa} | {loca} | {idf1} |\n"

with open("results/final_thesis_results_all.md", "w") as f:
    f.write("## Grand Unified Optimal Metrics Table\n")
    f.write(md)

print("\nSaved master table to results/final_thesis_results_all.md!")
