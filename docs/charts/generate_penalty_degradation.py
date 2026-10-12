import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

sns.set_theme(style="whitegrid")
out_dir = "/home/eelis/Desktop/Diplomityö/peopleTrackingWithLidar/docs_and_notes/charts"
csv_file = os.path.join(out_dir, "full_ablation_grid.csv")

def plot_degradation():
    df = pd.read_csv(csv_file)
    
    # Specific variants to plot
    target_variants = [
        'cp_kitchen_sink', 'hmfn_kitchen_sink', 'vn_kitchen_sink',
        'cp_swap', 'hmfn_swap', 'vn_swap'
    ]
    
    # Pretty names for the legend
    rename_map = {
        'cp_kitchen_sink': 'CenterPoint Multi-Modal',
        'hmfn_kitchen_sink': 'HMFN Multi-Modal',
        'vn_kitchen_sink': 'VoxelNeXt Multi-Modal',
        'cp_swap': 'CenterPoint Swap',
        'hmfn_swap': 'HMFN Swap',
        'vn_swap': 'VoxelNeXt Swap'
    }
    
    plot_df = df[df['Architecture_Variant'].isin(target_variants)].copy()
    
    # Replace internal names with pretty names
    plot_df['Architecture_Variant'] = plot_df['Architecture_Variant'].map(rename_map)
    
    # Group by Model and Reflectivity Penalty to get the mean across Dist/Far combinations
    grouped = plot_df.groupby(['Architecture_Variant', 'Reflectivity_Penalty_Weight'])['AMOTA_Pct'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # We will use 3 colors (one for each architecture) and 2 line styles (Multi-Modal vs Swap)
    colors = {
        'CenterPoint': '#1f77b4', # blue
        'HMFN': '#ff7f0e',        # orange
        'VoxelNeXt': '#2ca02c'    # green
    }
    
    for model in sorted(grouped['Architecture_Variant'].unique()):
        data = grouped[grouped['Architecture_Variant'] == model]
        
        # Determine color based on architecture
        base_arch = model.split(' ')[0]
        color = colors.get(base_arch, '#333333')
        
        # Determine line style (Swap = dashed, Multi-Modal = solid)
        is_swap = 'Swap' in model
        linestyle = '--' if is_swap else '-'
        marker = 's' if is_swap else 'o'
        
        ax.plot(data['Reflectivity_Penalty_Weight'], data['AMOTA_Pct'], 
                marker=marker, linestyle=linestyle, lw=3.0, markersize=9, 
                color=color, label=model, alpha=0.9)
        
    ax.set_title('AMOTA Degradation vs. Reflectivity Penalty', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Reflectivity Penalty Strictness (Weight)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tracking Performance (AMOTA %)', fontsize=12, fontweight='bold')
    
    # Clean up X axis ticks
    ax.set_xticks([0.0, 0.1, 0.3, 0.5])
    
    # Adjust Legend to sit outside
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=11, frameon=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '13_reflectivity_penalty_degradation.pdf'), bbox_inches='tight')
    print("Saved to 13_reflectivity_penalty_degradation.pdf")

if __name__ == '__main__':
    plot_degradation()
