import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys
import numpy as np

sns.set_theme(style="whitegrid")
out_dir = "/home/eelis/Desktop/Diplomityö/peopleTrackingWithLidar/docs_and_notes/charts"
data_path = os.path.join(out_dir, "full_ablation_grid.csv")

def plot_tracker_rigidity_multiple_models():
    df = pd.read_csv(data_path)
    
    # We select models that exhibit the tracker paradox at Reflectivity Weight 0
    selected_variants = ['cp_baseline', 'hmfn_baseline', 'vn_kitchen_sink']
    
    df_filtered = df[(df['Architecture_Variant'].isin(selected_variants)) & (df['Reflectivity_Penalty_Weight'] == 0.0)].copy()
    
    # Create a unified config label for tracker params
    df_filtered['Tracker Configuration'] = df_filtered.apply(lambda row: f"Dist={row['Distance_Threshold_m']:.0f}m, Far={row['Far_Score_Threshold']:.1f}", axis=1)
    
    # Create readable model names
    name_map = {
        'cp_baseline': 'CenterPoint Baseline',
        'hmfn_baseline': 'HMFN Baseline',
        'vn_kitchen_sink': 'VoxelNeXt MultiModal'
    }
    df_filtered['Model'] = df_filtered['Architecture_Variant'].map(name_map)
    
    # Sort for consistency
    df_filtered = df_filtered.sort_values(by=['Model', 'Distance_Threshold_m', 'Far_Score_Threshold'])
    
    # Apply a smart dodge to separate overlapping markers
    for idx in df_filtered.index:
        dist = df_filtered.loc[idx, 'Distance_Threshold_m']
        far = df_filtered.loc[idx, 'Far_Score_Threshold']
        
        # Offsets (dx is for IDF1, dy is for AMOTA)
        dx, dy = 0.0, 0.0
        # Increased dodge size to ensure markers don't touch
        if dist == 30 and far == 0.5:
            dx, dy = -0.15, 0.1
        elif dist == 30 and far == 0.6:
            dx, dy = 0.15, 0.1
        elif dist == 40 and far == 0.5:
            dx, dy = -0.15, -0.1
        elif dist == 40 and far == 0.6:
            dx, dy = 0.15, -0.1
            
        df_filtered.loc[idx, 'IDF1_Pct_plot'] = df_filtered.loc[idx, 'IDF1_Pct'] + dx
        df_filtered.loc[idx, 'AMOTA_Pct_plot'] = df_filtered.loc[idx, 'AMOTA_Pct'] + dy
            
    plt.figure(figsize=(10, 6.5))
    
    # Plot scatter points all at once to get a unified legend, using jittered coordinates
    scatter = sns.scatterplot(
        data=df_filtered,
        x='IDF1_Pct_plot',
        y='AMOTA_Pct_plot',
        hue='Model',
        style='Tracker Configuration',
        s=120, # Reduced marker size to prevent overlapping
        alpha=0.9,
        edgecolor="white", # white edge helps separate them visually
        palette="Dark2",
        linewidth=1.0
    )
    
    # Draw a line connecting the ORIGINAL (non-jittered) points for each model to show the true trade-off frontier
    colors = sns.color_palette("Dark2", n_colors=len(selected_variants))
    for idx, model in enumerate(df_filtered['Model'].unique()):
        model_data = df_filtered[df_filtered['Model'] == model].sort_values('IDF1_Pct')
        plt.plot(model_data['IDF1_Pct'], model_data['AMOTA_Pct'], color=colors[idx], alpha=0.4, linewidth=3, zorder=1)
        
        # Draw a tiny connecting line from the true coordinate to the jittered marker
        for _, row in df_filtered[df_filtered['Model'] == model].iterrows():
            plt.plot([row['IDF1_Pct'], row['IDF1_Pct_plot']], 
                     [row['AMOTA_Pct'], row['AMOTA_Pct_plot']], 
                     color=colors[idx], alpha=0.5, linewidth=1, linestyle='--')

    plt.xlabel('IDF1 (%) - Identity Continuity', fontsize=12, fontweight='bold')
    plt.ylabel('AMOTA (%) - Average Tracking Accuracy', fontsize=12, fontweight='bold')
    
    # Move legend outside the plot
    # Explicitly scale the markers down in the legend to reduce right-pane overlap
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, title_fontsize=12, borderaxespad=0., markerscale=0.8, labelspacing=0.8)
    
    plt.tight_layout()
    output_filename = '3_tracker_rigidity_tradeoff.pdf'
    plt.savefig(os.path.join(out_dir, output_filename))
    
    # Also save as PNG for quick viewing
    plt.savefig(os.path.join(out_dir, output_filename.replace('.pdf', '.png')))
    plt.close()
    print(f"Chart saved to {os.path.join(out_dir, output_filename)}")

if __name__ == '__main__':
    plot_tracker_rigidity_multiple_models()
