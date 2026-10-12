import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

sns.set_theme(style="whitegrid")
out_dir = "/home/eelis/Desktop/Diplomityö/peopleTrackingWithLidar/docs_and_notes/charts"
data_path = os.path.join(out_dir, "full_ablation_grid.csv")

def plot_normalized_divergence():
    df = pd.read_csv(data_path)
    
    # Real mAP scores provided by the user
    models_map = {
        'HMFN Baseline': 86.35,
        'HMFN Multi-Modal': 72.07,
        'CenterPoint Baseline': 76.65,
        'CenterPoint Multi-Modal': 82.76,
        'VoxelNeXt Baseline': 80.52,
        'VoxelNeXt Multi-Modal': 78.35
    }
    
    # Hardcoded HOTA scores provided by the user
    models_hota = {
        'HMFN Baseline': 36.59,
        'HMFN Multi-Modal': 37.61,
        'CenterPoint Baseline': 39.73,
        'CenterPoint Multi-Modal': 40.06, 
        'VoxelNeXt Baseline': 31.75,
        'VoxelNeXt Multi-Modal': 33.22
    }
    
    # Normalize the scores (Min-Max to 0-100)
    map_values = list(models_map.values())
    hota_values = list(models_hota.values())
    
    min_map, max_map = min(map_values), max(map_values)
    min_hota, max_hota = min(hota_values), max(hota_values)
    
    normalized_data = []
    for model in models_map.keys():
        norm_map = ((models_map[model] - min_map) / (max_map - min_map)) * 100
        norm_hota = ((models_hota[model] - min_hota) / (max_hota - min_hota)) * 100
        
        normalized_data.append({
            'Model': model,
            'Metric': 'Detection (mAP)',
            'Normalized Score': norm_map,
            'Raw': f"{models_map[model]:.1f}%"
        })
        normalized_data.append({
            'Model': model,
            'Metric': 'Tracking (HOTA)',
            'Normalized Score': norm_hota,
            'Raw': f"{models_hota[model]:.1f}%"
        })
        
    df_norm = pd.DataFrame(normalized_data)
    
    plt.figure(figsize=(10, 8))
    
    # Custom palette distinguishing baselines from decoupled
    palette = sns.color_palette("tab10", len(models_map))
    
    # Draw the slopegraph
    ax = sns.pointplot(
        data=df_norm,
        x='Metric',
        y='Normalized Score',
        hue='Model',
        palette=palette,
        markers='o',
        markersize=8,
        linewidth=2
    )
    
    # Add the title
    plt.title('Detection vs. Tracking Performance', fontsize=14, fontweight='bold', pad=15)
    
    # Clean up axes
    plt.ylabel('Relative Performance Score (0-100 Normalized)', fontsize=12, fontweight='bold')
    plt.xlabel('', fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Detection (mAP)', 'Tracking (HOTA)'], fontsize=12, fontweight='bold')
    
    # Legend outside
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=11)
    
    # Optional: annotate the raw scores next to the points
    for i, model in enumerate(models_map.keys()):
        # Left side (mAP)
        plt.text(-0.05, ((models_map[model] - min_map) / (max_map - min_map)) * 100, 
                 f"{models_map[model]:.1f}%", ha='right', va='center', fontsize=9)
        # Right side (HOTA)
        plt.text(1.05, ((models_hota[model] - min_hota) / (max_hota - min_hota)) * 100, 
                 f"{models_hota[model]:.1f}%", ha='left', va='center', fontsize=9)
                 
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '4_detection_tracking_divergence.pdf'))
    plt.close()
    print("Chart saved!")

if __name__ == '__main__':
    plot_normalized_divergence()
