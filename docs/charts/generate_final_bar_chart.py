import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

sns.set_theme(style="whitegrid")
out_dir = "/home/eelis/Desktop/Diplomityö/peopleTrackingWithLidar/docs_and_notes/charts"

def plot_final_comparison():
    # Data manually parsed from the final_ablation_metrics.md to ensure absolute accuracy
    data = [
        # CenterPoint
        {'Architecture': 'CenterPoint', 'Variant': 'Baseline (I)', 'AMOTA': 74.4},
        {'Architecture': 'CenterPoint', 'Variant': 'Decoupled (I, d)', 'AMOTA': 72.7},
        {'Architecture': 'CenterPoint', 'Variant': 'Enhanced (I, R)', 'AMOTA': 71.5},
        {'Architecture': 'CenterPoint', 'Variant': 'Kitchen Sink (I, R, d)', 'AMOTA': 68.1},
        {'Architecture': 'CenterPoint', 'Variant': 'Swap (R, d)', 'AMOTA': 0.0},
        
        # VoxelNeXt
        {'Architecture': 'VoxelNeXt', 'Variant': 'Baseline (I)', 'AMOTA': 38.3},
        {'Architecture': 'VoxelNeXt', 'Variant': 'Decoupled (I, d)', 'AMOTA': 70.1},
        {'Architecture': 'VoxelNeXt', 'Variant': 'Enhanced (I, R)', 'AMOTA': 0.0}, # Was not in top list or failed
        {'Architecture': 'VoxelNeXt', 'Variant': 'Kitchen Sink (I, R, d)', 'AMOTA': 71.1},
        {'Architecture': 'VoxelNeXt', 'Variant': 'Swap (R, d)', 'AMOTA': 24.5},
        
        # HMFN
        {'Architecture': 'HMFN', 'Variant': 'Baseline (I)', 'AMOTA': 71.7},
        {'Architecture': 'HMFN', 'Variant': 'Decoupled (I, d)', 'AMOTA': 65.7},
        {'Architecture': 'HMFN', 'Variant': 'Enhanced (I, R)', 'AMOTA': 71.9},
        {'Architecture': 'HMFN', 'Variant': 'Kitchen Sink (I, R, d)', 'AMOTA': 69.7},
        {'Architecture': 'HMFN', 'Variant': 'Swap (R, d)', 'AMOTA': 56.5},
        
        # PointPillars (Negative Control)
        {'Architecture': 'PointPillars (2D)', 'Variant': 'Baseline (I)', 'AMOTA': 0.0},
        {'Architecture': 'PointPillars (2D)', 'Variant': 'Decoupled (I, d)', 'AMOTA': 0.0},
        {'Architecture': 'PointPillars (2D)', 'Variant': 'Enhanced (I, R)', 'AMOTA': 0.0},
        {'Architecture': 'PointPillars (2D)', 'Variant': 'Kitchen Sink (I, R, d)', 'AMOTA': 0.0},
        {'Architecture': 'PointPillars (2D)', 'Variant': 'Swap (R, d)', 'AMOTA': 0.0},
    ]
    
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    
    # Custom academic color palette
    palette = sns.color_palette("muted", 5)
    
    # Grouped Bar Chart
    sns.barplot(data=df, x='Architecture', y='AMOTA', hue='Variant', palette=palette, ax=ax, edgecolor='black', linewidth=1)
    
    # Annotate values on top of bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}%', 
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, fontweight='bold',
                        xytext=(0, 5), textcoords='offset points')
    
    ax.set_title('Absolute Best AMOTA per Model Architecture & Feature Variant', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Model Architecture', fontsize=13, fontweight='bold')
    ax.set_ylabel('Peak AMOTA (%)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 85)
    
    # Draw a line showing the 'State of the Art' threshold
    ax.axhline(y=70.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(-0.4, 71.0, 'State-of-the-Art Baseline (>70%)', color='red', fontsize=11, fontweight='bold')
    
    ax.legend(title='Feature Engineering Strategy', title_fontsize='12', fontsize='11', loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '14_final_architecture_comparison.pdf'), bbox_inches='tight')
    print("Saved to 14_final_architecture_comparison.pdf")

if __name__ == '__main__':
    plot_final_comparison()
