import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

sns.set_theme(style="whitegrid")
out_dir = "/home/eelis/Desktop/Diplomityö/peopleTrackingWithLidar/docs_and_notes/charts"
csv_file = "/home/eelis/Desktop/Diplomityö/peopleTrackingWithLidar/docs_and_notes/charts/full_ablation_grid.csv"

def plot_macro_trends():
    df = pd.read_csv(csv_file)
    
    # Create a broad 'Variant Group' feature
    def categorize_variant(model_name):
        if 'decoupled' in model_name or 'baseline' in model_name:
            if 'pp_' in model_name:
                return '2D Architectures (PointPillars)'
            return 'Native Learning (Decoupled & Baseline)'
        elif 'swap' in model_name:
            if 'pp_' in model_name:
                return '2D Architectures (PointPillars)'
            return 'Forced Physics (Swap Models)'
        elif 'kitchen' in model_name:
            return 'Kitchen Sink'
        return 'Other'
        
    df['Variant_Group'] = df['Architecture_Variant'].apply(categorize_variant)
    
    # Filter out Kitchen Sink and 2D for this specific trend plot
    plot_df = df[df['Variant_Group'].isin(['Native Learning (Decoupled & Baseline)', 'Forced Physics (Swap Models)'])]
    
    # Group by Variant Group and Reflectivity Penalty
    grouped = plot_df.groupby(['Variant_Group', 'Reflectivity_Penalty_Weight'])[['AMOTA_Pct', 'IDF1_Pct']].mean().reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    groups = ['Native Learning (Decoupled & Baseline)', 'Forced Physics (Swap Models)']
    colors = ['#1f77b4', '#d62728']
    
    # Panel 1: AMOTA Trend
    for i, group in enumerate(groups):
        data = grouped[grouped['Variant_Group'] == group]
        axes[0].plot(data['Reflectivity_Penalty_Weight'], data['AMOTA_Pct'], 
                     marker='o', lw=3.5, markersize=10, color=colors[i], label=group)
        
    axes[0].set_title('Detection Volume (AMOTA) Macro-Trend', fontsize=14, fontweight='bold', pad=15)
    axes[0].set_ylabel('Mean AMOTA (%)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Reflectivity Penalty Strictness (x)', fontsize=12)
    axes[0].set_xticks([0.0, 0.1, 0.3, 0.5])
    axes[0].set_ylim(0, 80)
    
    # Panel 2: IDF1 Trend
    for i, group in enumerate(groups):
        data = grouped[grouped['Variant_Group'] == group]
        axes[1].plot(data['Reflectivity_Penalty_Weight'], data['IDF1_Pct'], 
                     marker='s', lw=3.5, markersize=10, color=colors[i], label=group)
                     
    axes[1].set_title('Identity Coherence (IDF1) Macro-Trend', fontsize=14, fontweight='bold', pad=15)
    axes[1].set_ylabel('Mean IDF1 (%)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Reflectivity Penalty Strictness (x)', fontsize=12)
    axes[1].set_xticks([0.0, 0.1, 0.3, 0.5])
    axes[1].set_ylim(10, 40)
    
    # Shared formatting
    for ax in axes:
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Single Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=2, fontsize=12, frameon=True)
    
    plt.suptitle('Universal Feature Trends Across All 3D Architectures', fontsize=16, fontweight='bold', y=1.18)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '11_variant_macro_trends.pdf'), bbox_inches='tight')

if __name__ == '__main__':
    plot_macro_trends()
