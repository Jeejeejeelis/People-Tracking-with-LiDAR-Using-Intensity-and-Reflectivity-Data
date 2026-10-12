import json
import os
from compute_pfsd_hota import compute_hota
from compute_pfsd_idf1 import compute_idf1
from io import StringIO
import sys

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio
        sys.stdout = self._stdout

def extract_metric(lines, key):
    for line in lines:
        if line.startswith(key + ":"):
            try:
                return float(line.split(":")[1].split()[0].strip())
            except:
                pass
    return 0.0

def generate_table():
    # Map model display names to their CURRENT BEST tracking result directories
    models = {
        "HMFN Baseline (I, t)": "results/hmfn_baseline_160/tracking_eval",
        "HMFN Decoupled (I, d, t)": "results/hmfn_decoupled_160/tracking_eval",
        "HMFN Swap (R, d, t)": "results/hmfn_swap_160/tracking_eval",
        
        "CenterPoint Baseline (I, t)": "results/cp_baseline_160/tracking_eval",
        "CenterPoint Decoupled (I, d, t)": "results/pfsd_centerpoint_decoupled/tracking_eval",
        "CenterPoint Swap (R, d, t)": "results/swap_tracking_verify",
        
        "VoxelNeXt Baseline (I, t)": "results/vn_baseline_160/tracking_eval",
        "VoxelNeXt Decoupled (I, d, t)": "results/pfsd_voxelnext_decoupled/tracking_eval",
        "VoxelNeXt Swap (R, d, t)": "results/vn_swap_160/tracking_eval"
    }
    
    gt_path = "PFSD/data/nuscenes/v1.0-mini/pfsd_tracking_gt.json"
    
    print("| **Model / Configuration** | **AMOTA** | **HOTA** | **DetA** | **AssA** | **LocA** | **IDF1** |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for name, path in models.items():
        results_file = os.path.join(path, "tracking_results.json")
        summary_file = os.path.join(path, "metrics_summary.json")
        
        if not os.path.exists(results_file) or not os.path.exists(summary_file):
            print(f"| {name} | (Missing Data) | - | - | - | - | - |")
            continue
            
        with open(summary_file, 'r') as f:
            data = json.load(f)
            amota = data.get('amota', 0.0)
            
        with Capturing() as output_hota:
            compute_hota(results_file, gt_path)
        hota = extract_metric(output_hota, "HOTA")
        deta = extract_metric(output_hota, "DetA")
        assa = extract_metric(output_hota, "AssA")
        loca = extract_metric(output_hota, "LocA")
        
        with Capturing() as output_idf1:
            compute_idf1(results_file, gt_path)
        idf1 = extract_metric(output_idf1, "IDF1")
        
        print(f"| {name} | {amota*100:.1f}% | {hota*100:.1f}% | {deta*100:.1f}% | {assa*100:.1f}% | {loca*100:.1f}% | {idf1*100:.1f}% |")

if __name__ == '__main__':
    generate_table()
