import pandas as pd
import os

report_file = "/home/eelis/Desktop/Diplomityö/peopleTrackingWithLidar/results/ablation_tuning/final_report.md"
out_csv = "/home/eelis/Desktop/Diplomityö/peopleTrackingWithLidar/docs_and_notes/charts/full_ablation_grid.csv"

def export_data():
    with open(report_file, 'r') as f:
        lines = f.readlines()
        
    data = []
    current_model = None
    
    for line in lines:
        if line.startswith('## Model:'):
            current_model = line.split('## Model:')[1].strip()
        elif current_model and line.startswith('|') and 'Dist Thresh' not in line and '---' not in line:
            parts = [p.strip() for p in line.split('|')[1:-1]]
            if len(parts) >= 10:
                try:
                    # Clean and parse the fields
                    dist = float(parts[0].replace('m', ''))
                    far = float(parts[1])
                    refl = float(parts[2].replace('x', ''))
                    amota = float(parts[3].replace('*', '')) * 100
                    hota = float(parts[4]) * 100 if parts[4] else 0.0
                    idf1 = float(parts[5]) * 100 if parts[5] else 0.0
                    amotp = float(parts[6]) if parts[6] else 0.0
                    recall = float(parts[7].replace('*', '')) * 100 if parts[7] else 0.0
                    fp = int(parts[8].replace(',', '')) if parts[8] else 0
                    tp = int(parts[9].replace(',', '')) if parts[9] else 0
                    
                    data.append({
                        'Architecture_Variant': current_model,
                        'Distance_Threshold_m': dist,
                        'Far_Score_Threshold': far,
                        'Reflectivity_Penalty_Weight': refl,
                        'AMOTA_Pct': amota,
                        'HOTA_Pct': hota,
                        'IDF1_Pct': idf1,
                        'AMOTP': amotp,
                        'Recall_Pct': recall,
                        'False_Positives': fp,
                        'True_Positives': tp
                    })
                except ValueError as e:
                    pass
                    
    df = pd.DataFrame(data)
    df.to_csv(out_csv, index=False)
    print(f"Successfully exported {len(df)} rows to {out_csv}")

if __name__ == '__main__':
    export_data()
