import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

sns.set_theme(style="whitegrid")
out_dir = "/home/eelis/Desktop/Diplomityö/peopleTrackingWithLidar/docs_and_notes/charts"

data = [
    # CenterPoint
    {'Architecture': 'CenterPoint', 'Variant': 'Baseline (I)', 'AMOTA': 74.4, 'HOTA': 39.18, 'IDF1': 37.94},
    {'Architecture': 'CenterPoint', 'Variant': 'Decoupled (I, d)', 'AMOTA': 72.7, 'HOTA': 37.99, 'IDF1': 36.51},
    {'Architecture': 'CenterPoint', 'Variant': 'Enhanced (I, R)', 'AMOTA': 71.5, 'HOTA': 36.78, 'IDF1': 35.06},
    {'Architecture': 'CenterPoint', 'Variant': 'Kitchen Sink (I, R, d)', 'AMOTA': 68.1, 'HOTA': 40.06, 'IDF1': 39.12},
    {'Architecture': 'CenterPoint', 'Variant': 'Swap (R, d)', 'AMOTA': 0.0, 'HOTA': 0.0, 'IDF1': 0.0},
    
    # VoxelNeXt
    {'Architecture': 'VoxelNeXt', 'Variant': 'Baseline (I)', 'AMOTA': 69.9, 'HOTA': 31.75, 'IDF1': 30.90},
    {'Architecture': 'VoxelNeXt', 'Variant': 'Decoupled (I, d)', 'AMOTA': 70.1, 'HOTA': 32.04, 'IDF1': 31.54},
    {'Architecture': 'VoxelNeXt', 'Variant': 'Enhanced (I, R)', 'AMOTA': 69.8, 'HOTA': 31.55, 'IDF1': 30.60}, 
    {'Architecture': 'VoxelNeXt', 'Variant': 'Kitchen Sink (I, R, d)', 'AMOTA': 71.1, 'HOTA': 33.22, 'IDF1': 32.41},
    {'Architecture': 'VoxelNeXt', 'Variant': 'Swap (R, d)', 'AMOTA': 24.5, 'HOTA': 20.13, 'IDF1': 18.54},
    
    # HMFN
    {'Architecture': 'HMFN', 'Variant': 'Baseline (I)', 'AMOTA': 71.7, 'HOTA': 36.59, 'IDF1': 34.96},
    {'Architecture': 'HMFN', 'Variant': 'Decoupled (I, d)', 'AMOTA': 65.7, 'HOTA': 34.88, 'IDF1': 34.07},
    {'Architecture': 'HMFN', 'Variant': 'Enhanced (I, R)', 'AMOTA': 71.9, 'HOTA': 37.06, 'IDF1': 37.20},
    {'Architecture': 'HMFN', 'Variant': 'Kitchen Sink (I, R, d)', 'AMOTA': 69.7, 'HOTA': 37.61, 'IDF1': 37.27},
    {'Architecture': 'HMFN', 'Variant': 'Swap (R, d)', 'AMOTA': 56.5, 'HOTA': 31.46, 'IDF1': 32.44},
    
    # PointPillars (Negative Control)
    {'Architecture': 'PointPillars (2D)', 'Variant': 'Baseline (I)', 'AMOTA': 0.0, 'HOTA': 0.0, 'IDF1': 0.0},
    {'Architecture': 'PointPillars (2D)', 'Variant': 'Decoupled (I, d)', 'AMOTA': 0.0, 'HOTA': 0.0, 'IDF1': 0.0},
    {'Architecture': 'PointPillars (2D)', 'Variant': 'Enhanced (I, R)', 'AMOTA': 0.0, 'HOTA': 0.0, 'IDF1': 0.0},
    {'Architecture': 'PointPillars (2D)', 'Variant': 'Kitchen Sink (I, R, d)', 'AMOTA': 0.0, 'HOTA': 0.0, 'IDF1': 0.0},
    {'Architecture': 'PointPillars (2D)', 'Variant': 'Swap (R, d)', 'AMOTA': 0.0, 'HOTA': 7.77, 'IDF1': 6.93},
]

df = pd.DataFrame(data)

def plot_metric(metric, filename, title, ylim, threshold=None, threshold_text=None):
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    palette = sns.color_palette("muted", 5)
    
    sns.barplot(data=df, x='Architecture', y=metric, hue='Variant', palette=palette, ax=ax, edgecolor='black', linewidth=1)
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}%', 
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, fontweight='bold',
                        xytext=(0, 5), textcoords='offset points')
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Model Architecture', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'Peak {metric} (%)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, ylim)
    
    if threshold is not None:
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(-0.4, threshold + 0.5, threshold_text, color='red', fontsize=11, fontweight='bold')
    
    ax.legend(title='Feature Engineering Strategy', title_fontsize='12', fontsize='11', loc='upper right')
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved to {filename}")

if __name__ == '__main__':
    # AMOTA
    plot_metric('AMOTA', '14_final_architecture_comparison.pdf', 'Absolute Best AMOTA per Model Architecture & Feature Variant', 85, threshold=70.0, threshold_text='State-of-the-Art Baseline (>70%)')
    
    # HOTA
    plot_metric('HOTA', '15_final_architecture_comparison_hota.pdf', 'Absolute Best HOTA per Model Architecture & Feature Variant', 45, threshold=35.0, threshold_text='High Trajectory Continuity (>35%)')
    
    # IDF1
    plot_metric('IDF1', '16_final_architecture_comparison_idf1.pdf', 'Absolute Best IDF1 (Identity F1-Score) per Model Architecture', 45, threshold=35.0, threshold_text='High Identity Preservation (>35%)')
