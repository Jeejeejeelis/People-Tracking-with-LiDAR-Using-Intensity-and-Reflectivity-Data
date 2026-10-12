import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

sns.set_theme(style="whitegrid")
out_dir = "/home/eelis/Desktop/Diplomityö/peopleTrackingWithLidar/docs_and_notes/charts"

data = [
    # CenterPoint
    {'Architecture': 'CenterPoint', 'Variant': 'Baseline', 'Metric': 'AMOTA', 'Score': 74.4},
    {'Architecture': 'CenterPoint', 'Variant': 'Baseline', 'Metric': 'IDF1', 'Score': 37.94},
    {'Architecture': 'CenterPoint', 'Variant': 'Decoupled', 'Metric': 'AMOTA', 'Score': 72.7},
    {'Architecture': 'CenterPoint', 'Variant': 'Decoupled', 'Metric': 'IDF1', 'Score': 36.51},
    {'Architecture': 'CenterPoint', 'Variant': 'Enhanced', 'Metric': 'AMOTA', 'Score': 71.5},
    {'Architecture': 'CenterPoint', 'Variant': 'Enhanced', 'Metric': 'IDF1', 'Score': 35.06},
    {'Architecture': 'CenterPoint', 'Variant': 'Multi-Modal', 'Metric': 'AMOTA', 'Score': 68.1},
    {'Architecture': 'CenterPoint', 'Variant': 'Multi-Modal', 'Metric': 'IDF1', 'Score': 39.12},
    {'Architecture': 'CenterPoint', 'Variant': 'Swap', 'Metric': 'AMOTA', 'Score': 0.0},
    {'Architecture': 'CenterPoint', 'Variant': 'Swap', 'Metric': 'IDF1', 'Score': 0.0},
    
    # HMFN
    {'Architecture': 'HMFN', 'Variant': 'Baseline', 'Metric': 'AMOTA', 'Score': 71.7},
    {'Architecture': 'HMFN', 'Variant': 'Baseline', 'Metric': 'IDF1', 'Score': 34.96},
    {'Architecture': 'HMFN', 'Variant': 'Decoupled', 'Metric': 'AMOTA', 'Score': 65.7},
    {'Architecture': 'HMFN', 'Variant': 'Decoupled', 'Metric': 'IDF1', 'Score': 34.07},
    {'Architecture': 'HMFN', 'Variant': 'Enhanced', 'Metric': 'AMOTA', 'Score': 71.9},
    {'Architecture': 'HMFN', 'Variant': 'Enhanced', 'Metric': 'IDF1', 'Score': 37.20},
    {'Architecture': 'HMFN', 'Variant': 'Multi-Modal', 'Metric': 'AMOTA', 'Score': 69.7},
    {'Architecture': 'HMFN', 'Variant': 'Multi-Modal', 'Metric': 'IDF1', 'Score': 37.27},
    {'Architecture': 'HMFN', 'Variant': 'Swap', 'Metric': 'AMOTA', 'Score': 56.5},
    {'Architecture': 'HMFN', 'Variant': 'Swap', 'Metric': 'IDF1', 'Score': 32.44},
    
    # VoxelNeXt
    {'Architecture': 'VoxelNeXt', 'Variant': 'Baseline', 'Metric': 'AMOTA', 'Score': 69.9},
    {'Architecture': 'VoxelNeXt', 'Variant': 'Baseline', 'Metric': 'IDF1', 'Score': 30.90},
    {'Architecture': 'VoxelNeXt', 'Variant': 'Decoupled', 'Metric': 'AMOTA', 'Score': 70.1},
    {'Architecture': 'VoxelNeXt', 'Variant': 'Decoupled', 'Metric': 'IDF1', 'Score': 31.54},
    {'Architecture': 'VoxelNeXt', 'Variant': 'Enhanced', 'Metric': 'AMOTA', 'Score': 69.8},
    {'Architecture': 'VoxelNeXt', 'Variant': 'Enhanced', 'Metric': 'IDF1', 'Score': 30.60}, 
    {'Architecture': 'VoxelNeXt', 'Variant': 'Multi-Modal', 'Metric': 'AMOTA', 'Score': 71.1},
    {'Architecture': 'VoxelNeXt', 'Variant': 'Multi-Modal', 'Metric': 'IDF1', 'Score': 32.41},
    {'Architecture': 'VoxelNeXt', 'Variant': 'Swap', 'Metric': 'AMOTA', 'Score': 24.5},
    {'Architecture': 'VoxelNeXt', 'Variant': 'Swap', 'Metric': 'IDF1', 'Score': 18.54}
]

df = pd.DataFrame(data)

def plot_grouped_bars():
    # We will create a 3-panel chart (one for each architecture)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300, sharey=True)
    
    architectures = ['CenterPoint', 'HMFN', 'VoxelNeXt']
    palette = {'AMOTA': '#1f77b4', 'IDF1': '#ff7f0e'}
    
    for idx, arch in enumerate(architectures):
        ax = axes[idx]
        subset = df[df['Architecture'] == arch]
        
        sns.barplot(data=subset, x='Variant', y='Score', hue='Metric', 
                    palette=palette, ax=ax, edgecolor='black', linewidth=1)
        
        # Annotate bars
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}', 
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', fontsize=9, fontweight='bold',
                            xytext=(0, 5), textcoords='offset points')
        
        # State-of-the-Art Baseline Markers
        ax.axhline(y=70.0, color='#d62728', linestyle='--', linewidth=2.5, alpha=1.0, zorder=0)
        if idx == 2:
            ax.text(4.5, 71.5, '>70%', color='#d62728', fontsize=12, fontweight='bold', zorder=5)
            
        ax.axhline(y=35.0, color='#2ca02c', linestyle='--', linewidth=2.5, alpha=1.0, zorder=0)
        if idx == 2:
            ax.text(4.5, 36.5, '>35%', color='#2ca02c', fontsize=12, fontweight='bold', zorder=5)
            
        ax.set_title(arch, fontsize=15, fontweight='bold', pad=15)
        ax.set_xlabel('')
        ax.set_xticks(range(len(subset['Variant'].unique())))
        ax.set_xticklabels(subset['Variant'].unique(), rotation=30, ha='right', fontsize=11, fontweight='bold')
        
        if idx == 0:
            ax.set_ylabel('Performance Score (%)', fontsize=13, fontweight='bold')
        else:
            ax.set_ylabel('')
            
        # Remove individual legends to create one global legend
        ax.get_legend().remove()
        
    ax.set_ylim(0, 85)
    
    # Global legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=12, frameon=True)
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, '14_final_grouped_bar_comparison.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    plot_grouped_bars()
