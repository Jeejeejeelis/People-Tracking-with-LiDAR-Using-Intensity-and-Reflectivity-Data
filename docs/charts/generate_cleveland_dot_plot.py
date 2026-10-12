import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

sns.set_theme(style="whitegrid")
out_dir = "/home/eelis/Desktop/Diplomityö/peopleTrackingWithLidar/docs_and_notes/charts"

# The data from the LaTeX table (excluding PointPillars)
data = [
    # CenterPoint
    {'Architecture': 'CenterPoint', 'Variant': 'Baseline (I)', 'AMOTA': 74.4, 'IDF1': 37.94},
    {'Architecture': 'CenterPoint', 'Variant': 'Decoupled (I, d)', 'AMOTA': 72.7, 'IDF1': 36.51},
    {'Architecture': 'CenterPoint', 'Variant': 'Enhanced (I, R)', 'AMOTA': 71.5, 'IDF1': 35.06},
    {'Architecture': 'CenterPoint', 'Variant': 'Kitchen Sink (I, R, d)', 'AMOTA': 68.1, 'IDF1': 39.12},
    {'Architecture': 'CenterPoint', 'Variant': 'Swap (R, d)', 'AMOTA': 0.0, 'IDF1': 0.0},
    
    # HMFN
    {'Architecture': 'HMFN', 'Variant': 'Baseline (I)', 'AMOTA': 71.7, 'IDF1': 34.96},
    {'Architecture': 'HMFN', 'Variant': 'Decoupled (I, d)', 'AMOTA': 65.7, 'IDF1': 34.07},
    {'Architecture': 'HMFN', 'Variant': 'Enhanced (I, R)', 'AMOTA': 71.9, 'IDF1': 37.20},
    {'Architecture': 'HMFN', 'Variant': 'Kitchen Sink (I, R, d)', 'AMOTA': 69.7, 'IDF1': 37.27},
    {'Architecture': 'HMFN', 'Variant': 'Swap (R, d)', 'AMOTA': 56.5, 'IDF1': 32.44},
    
    # VoxelNeXt
    {'Architecture': 'VoxelNeXt', 'Variant': 'Baseline (I)', 'AMOTA': 69.9, 'IDF1': 30.90},
    {'Architecture': 'VoxelNeXt', 'Variant': 'Decoupled (I, d)', 'AMOTA': 70.1, 'IDF1': 31.54},
    {'Architecture': 'VoxelNeXt', 'Variant': 'Enhanced (I, R)', 'AMOTA': 69.8, 'IDF1': 30.60}, 
    {'Architecture': 'VoxelNeXt', 'Variant': 'Kitchen Sink (I, R, d)', 'AMOTA': 71.1, 'IDF1': 32.41},
    {'Architecture': 'VoxelNeXt', 'Variant': 'Swap (R, d)', 'AMOTA': 24.5, 'IDF1': 18.54}
]

df = pd.DataFrame(data)

# Create a combined label for the Y-axis
df['Model_Variant'] = df['Architecture'] + " - " + df['Variant']

def plot_cleveland():
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    
    # We will plot lines connecting the two dots for each model, then plot the dots.
    # The models will be plotted on the Y-axis (so we reverse it to have CenterPoint at the top)
    y_positions = np.arange(len(df))[::-1]
    
    # Draw horizontal lines connecting 0 to AMOTA/IDF1 to guide the eye
    for i, y in enumerate(y_positions):
        # Determine background striping for readability
        if i % 5 == 0:
            ax.axhspan(y-0.5, y+0.5, color='gray', alpha=0.05, zorder=0)
            
        amota = df.iloc[i]['AMOTA']
        idf1 = df.iloc[i]['IDF1']
        
        # Connect the two points
        ax.plot([idf1, amota], [y, y], color='gray', linestyle='-', alpha=0.4, zorder=1)
        
        # Plot the dots
        ax.scatter(amota, y, color='#1f77b4', s=120, zorder=3, edgecolors='black', linewidth=0.5)
        ax.scatter(idf1, y, color='#ff7f0e', s=120, zorder=3, edgecolors='black', linewidth=0.5)
        
        # Annotate the values next to the dots (only if they are > 0)
        if amota > 0:
            ax.text(amota + 1.5, y, f"{amota:.1f}%", va='center', ha='left', fontsize=9, color='#1f77b4', fontweight='bold')
        if idf1 > 0:
            ax.text(idf1 - 1.5, y, f"{idf1:.1f}%", va='center', ha='right', fontsize=9, color='#d95f02', fontweight='bold')
            
    # Set Y-axis labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df['Model_Variant'], fontsize=11, fontweight='bold')
    
    # Draw State-of-the-Art Baseline Markers
    ax.axvline(x=70.0, color='#1f77b4', linestyle='--', alpha=0.7, zorder=0)
    ax.text(70.5, y_positions[-1] - 0.8, 'SOTA Detection Baseline (>70%)', color='#1f77b4', fontsize=10, fontweight='bold', ha='left')
    
    ax.axvline(x=35.0, color='#ff7f0e', linestyle='--', alpha=0.7, zorder=0)
    ax.text(35.5, y_positions[-1] - 0.8, 'SOTA Tracking Baseline (>35%)', color='#d95f02', fontsize=10, fontweight='bold', ha='left')
    
    # Formatting
    ax.set_xlim(-5, 85)
    ax.set_ylim(-1, len(df))
    ax.set_xlabel('Metric Performance (%)', fontsize=13, fontweight='bold')
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='AMOTA (Detection Biased)', markerfacecolor='#1f77b4', markersize=12, markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', label='IDF1 (Tracking Biased)', markerfacecolor='#ff7f0e', markersize=12, markeredgecolor='black')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.01, 0.99), fontsize=11, frameon=True)
    
    plt.title('Absolute Best Performance by Architecture and Feature Variant', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, '14_final_cleveland_comparison.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    plot_cleveland()
