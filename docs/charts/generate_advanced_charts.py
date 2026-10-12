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
                    amota = float(parts[3].replace('*', ''))
                    hota = float(parts[4]) if parts[4] else 0.0
                    idf1 = float(parts[5]) if parts[5] else 0.0
                    dist = float(parts[0].replace('m',''))
                    far = float(parts[1])
                    refl = float(parts[2].replace('x', ''))
                    data.append({'Model': current_model, 'Dist': dist, 'Far': far, 'Refl': refl, 'AMOTA': amota, 'HOTA': hota, 'IDF1': idf1})
                except ValueError:
                    pass
    return pd.DataFrame(data)

def plot_iso_hota():
    df = parse_data()
    # Let's filter to cp_baseline, refl=0.0 to just show the trade-off of Dist/Far Score
    plot_df = df[(df['Model'] == 'cp_baseline') & (df['Refl'] == 0.0)]
    
    # 1. Define Grid for Iso-HOTA Contours
    amota_grid = np.linspace(0.65, 0.80, 500)
    idf1_grid = np.linspace(0.30, 0.45, 500)
    AMOTA_m, IDF1_m = np.meshgrid(amota_grid, idf1_grid)
    # Using HOTA proxy: HOTA ≈ sqrt(AMOTA * IDF1)
    HOTA_proxy = np.sqrt(AMOTA_m * IDF1_m)

    plt.figure(figsize=(9, 7), dpi=300)
    contours = plt.contour(AMOTA_m, IDF1_m, HOTA_proxy, levels=np.arange(0.1, 0.9, 0.005), colors='gray', linestyles='dashed', alpha=0.5)
    plt.clabel(contours, inline=True, fontsize=9, fmt='HOTA≈%.3f')

    # 2. Plot tracker data points
    colors = sns.color_palette("viridis", len(plot_df))
    
    for i, row in plot_df.iterrows():
        label_str = f"Dist={row['Dist']}m, Far={row['Far']}"
        # Highlight the extreme points
        if row['Dist'] == 40.0 and row['Far'] == 0.5:
            # High AMOTA, Loose
            plt.scatter([row['AMOTA']], [row['IDF1']], color='#d62728', s=200, label='Loose Tracking (Max AMOTA)\nDist=40m, Far=0.5', zorder=5, edgecolor='black', linewidth=1.5)
            pt_loose = (row['AMOTA'], row['IDF1'])
        elif row['Dist'] == 30.0 and row['Far'] == 0.6:
            # High IDF1, Strict
            plt.scatter([row['AMOTA']], [row['IDF1']], color='#1f77b4', s=200, label='Strict Tracking (Max IDF1)\nDist=30m, Far=0.6', zorder=5, marker='s', edgecolor='black', linewidth=1.5)
            pt_strict = (row['AMOTA'], row['IDF1'])
        else:
            plt.scatter([row['AMOTA']], [row['IDF1']], color='gray', s=80, alpha=0.6)

    # 3. Connect with a directional trajectory arrow
    plt.annotate('', xy=pt_strict, xytext=pt_loose,
                 arrowprops=dict(arrowstyle="->", color='black', lw=2, mutation_scale=20, ls='dotted'))

    # Add text to explain the arrow
    plt.text((pt_strict[0]+pt_loose[0])/2, (pt_strict[1]+pt_loose[1])/2 + 0.001, 
             'Increasing Heuristic Strictness\n(Trades Volume for Purity)', 
             ha='center', va='bottom', fontsize=10, rotation=15)

    plt.xlabel('Detection Volume (AMOTA)', fontsize=12, fontweight='bold')
    plt.ylabel('Trajectory Identity Coherence (IDF1)', fontsize=12, fontweight='bold')
    plt.title('The Zero-Sum Tracking Paradox (CenterPoint Baseline):\nIso-HOTA Metric Plane', fontsize=14, fontweight='bold')
    
    # Dynamic axis limits
    plt.xlim(0.70, 0.76)
    plt.ylim(0.36, 0.40)
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower left', fontsize=10, frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '7_iso_hota_metric_plane.pdf'))
    plt.close()

def plot_dual_axis_crossover():
    df = parse_data()
    # Use vn_swap to show the crossover where strictness (refl penalty) trades AMOTA for IDF1
    # We will average across dist/far parameters to isolate Refl Weight
    plot_df = df[df['Model'] == 'vn_swap'].groupby('Refl').mean(numeric_only=True).reset_index()
    
    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=300)
    
    color1 = '#1f77b4'
    ax1.set_xlabel('Reflectivity Penalty Strictness (Refl Weight)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Detection Volume (AMOTA %)', color=color1, fontsize=12, fontweight='bold')
    ax1.plot(plot_df['Refl'], plot_df['AMOTA']*100, color=color1, marker='o', linewidth=3, markersize=10, label='AMOTA')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(plot_df['Refl'])
    ax1.set_xticklabels([f"{x}x" for x in plot_df['Refl']])
    
    ax2 = ax1.twinx()  
    color2 = '#d62728'
    ax2.set_ylabel('Trajectory Identity Coherence (IDF1 %)', color=color2, fontsize=12, fontweight='bold')  
    ax2.plot(plot_df['Refl'], plot_df['IDF1']*100, color=color2, marker='s', linewidth=3, markersize=10, label='IDF1')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Title
    plt.title('Dual-Axis Crossover: VoxelNeXt Swap\nStrict Physics Penalty Purges Noise at the Expense of Volume', fontsize=13, fontweight='bold', pad=20)
    
    # Legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '8_dual_axis_crossover.pdf'))
    plt.close()

if __name__ == '__main__':
    plot_iso_hota()
    plot_dual_axis_crossover()
    print("Advanced NotebookLM-style charts generated successfully!")
