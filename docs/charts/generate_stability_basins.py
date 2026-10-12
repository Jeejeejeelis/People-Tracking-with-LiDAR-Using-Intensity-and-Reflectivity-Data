import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

sns.set_theme(style="whitegrid")
out_dir = "/home/eelis/Desktop/Diplomityö/peopleTrackingWithLidar/docs_and_notes/charts"
report_file = "/home/eelis/Desktop/Diplomityö/peopleTrackingWithLidar/results/ablation_tuning/final_report.md"

def parse_data():
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
                    amota = float(parts[3].replace('*', '')) * 100
                    dist = float(parts[0].replace('m',''))
                    far = float(parts[1])
                    refl = float(parts[2].replace('x', ''))
                    data.append({'Model': current_model, 'Dist': dist, 'Far': far, 'Refl': refl, 'AMOTA': amota})
                except ValueError:
                    pass
    return pd.DataFrame(data)

def plot_stability_basins():
    df = parse_data()
    
    # Target models
    m1 = 'cp_decoupled'
    m2 = 'hmfn_swap'
    
    # Filter and reshape data for heatmaps
    def get_heatmap_matrix(model_name):
        subset = df[df['Model'] == model_name].copy()
        subset['Config'] = "Dist=" + subset['Dist'].astype(int).astype(str) + "m, Far=" + subset['Far'].astype(str)
        pivot = subset.pivot(index='Config', columns='Refl', values='AMOTA')
        return pivot

    pivot_m1 = get_heatmap_matrix(m1)
    pivot_m2 = get_heatmap_matrix(m2)
    
    # Sort index so Dist=30 comes before Dist=40
    pivot_m1 = pivot_m1.sort_index()
    pivot_m2 = pivot_m2.sort_index()

    # Find 90% peak threshold for contour
    peak_m1 = pivot_m1.values.max()
    peak_m2 = pivot_m2.values.max()
    thresh_m1 = peak_m1 * 0.90
    thresh_m2 = peak_m2 * 0.90
    
    fig = plt.figure(figsize=(14, 10), dpi=300)
    
    # GridSpec to arrange 2 heatmaps on top, 1 line plot on bottom
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2, height_ratios=[1.2, 1], hspace=0.35)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    
    # Helper to plot heatmap with 90% stability contour
    def draw_heatmap(ax, pivot, title, thresh):
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", 
                    vmin=0, vmax=75, cbar=True, cbar_kws={'label': 'AMOTA (%)'}, ax=ax)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel('Reflectivity Penalty Weight (x)', fontsize=11)
        ax.set_ylabel('Tracker Configuration', fontsize=11)
        
        # Draw explicit boundaries around the stability basin (cells >= thresh)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                if pivot.iloc[i, j] >= thresh:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=3, ls='--'))

    draw_heatmap(ax1, pivot_m1, "CenterPoint Decoupled\n(Native 3D Feature Learning)", thresh_m1)
    draw_heatmap(ax2, pivot_m2, "HMFN Swap\n(Forced Physics Equation)", thresh_m2)
    
    # Custom legend for the red dashed box
    import matplotlib.patches as mpatches
    basin_patch = mpatches.Patch(facecolor='none', edgecolor='red', lw=2, ls='--', label='Stability Basin Area ($>= 90\%$ Peak)')
    ax2.legend(handles=[basin_patch], loc='upper right', bbox_to_anchor=(1.0, 1.25), fontsize=10)

    # 1D Cross-Sectional Profile Line
    slice_config = "Dist=40m, Far=0.5"
    slice_m1 = pivot_m1.loc[slice_config]
    slice_m2 = pivot_m2.loc[slice_config]
    
    ax3.plot(slice_m1.index, slice_m1.values, marker='o', lw=3, markersize=10, color='#1f77b4', label='CenterPoint Decoupled (Mode Collapse)')
    ax3.plot(slice_m2.index, slice_m2.values, marker='s', lw=3, markersize=10, color='#d62728', label='HMFN Swap (Resilient Plateau)')
    
    ax3.set_title(f'1D Cross-Sectional Profile (at {slice_config}):\nStep-Function Cliff vs. Resilient Plateau', fontsize=13, fontweight='bold', pad=15)
    ax3.set_xlabel('Reflectivity Penalty Weight (x)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('AMOTA (%)', fontsize=11, fontweight='bold')
    ax3.set_xticks(slice_m1.index)
    ax3.set_xticklabels([f"{x}x" for x in slice_m1.index])
    ax3.set_ylim(0, 80)
    ax3.legend(loc='lower left', fontsize=11, frameon=True)
    
    plt.suptitle('Hyperparameter Stability Basins & Mode Collapse\nQuantifying Tracker Robustness', fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '10_hyperparameter_stability_basins.pdf'), bbox_inches='tight')
    print("Saved stability basin chart to", os.path.join(out_dir, '10_hyperparameter_stability_basins.pdf'))

if __name__ == '__main__':
    plot_stability_basins()
