import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

sns.set_theme(style="whitegrid")
out_dir = "/home/eelis/Desktop/Diplomityö/peopleTrackingWithLidar/docs_and_notes/charts"

def plot_hota_slopegraph():
    # Data from Phase 12 of the thesis notes (DetA, AssA, LocA divergence)
    # CenterPoint Decoupled (AI learns physics natively) vs CenterPoint Swap (Forced physics equation)
    metrics = ['Detection Accuracy\n(DetA %)', 'Association Accuracy\n(AssA %)', 'Localization Accuracy\n(LocA %)']
    
    # Decoupled: Lower DetA, but much higher AssA and LocA
    decoupled_scores = [64.3, 56.5, 82.0]
    
    # Swap: Higher DetA, but crashes AssA and LocA due to background ghosting and reflective box shifting
    swap_scores = [65.6, 38.2, 73.1]
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    x = [1, 2, 3]
    
    # Plot the lines
    ax.plot(x, decoupled_scores, marker='o', markersize=10, linewidth=3.5, 
            color='#1f77b4', label='CenterPoint Decoupled\n(Stable Localization & Identity)', zorder=5)
    
    ax.plot(x, swap_scores, marker='s', markersize=10, linewidth=3.5, 
            color='#d62728', label='CenterPoint Swap\n(High DetA, Crashing AssA/LocA)', zorder=5)
    
    # Annotate the values
    for i in range(len(x)):
        # Decoupled
        ax.text(x[i] - 0.05 if i == 0 else x[i] + 0.05, decoupled_scores[i] + (1.5 if i==0 else 1.0), 
                f"{decoupled_scores[i]}%", color='#1f77b4', fontweight='bold', fontsize=11, 
                ha='right' if i == 0 else 'left')
        
        # Swap
        ax.text(x[i] - 0.05 if i == 0 else x[i] + 0.05, swap_scores[i] - (1.5 if i==0 else 1.5), 
                f"{swap_scores[i]}%", color='#d62728', fontweight='bold', fontsize=11, 
                ha='right' if i == 0 else 'left')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(20, 95)
    ax.set_ylabel('Metric Score (%)', fontsize=12, fontweight='bold')
    
    # Draw vertical lines for the axes to make it look like parallel coordinates
    for pos in x:
        ax.axvline(x=pos, color='gray', linestyle='-', alpha=0.3, zorder=1)
        
    plt.title('HOTA Sub-Component Divergence (Parallel Coordinates):\nThe "X-Shape" Trade-off of Physical Reflectivity', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=11, frameon=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '9_hota_subcomponent_slopegraph.pdf'), bbox_inches='tight')
    print("Saved slopegraph to", os.path.join(out_dir, '9_hota_subcomponent_slopegraph.pdf'))

if __name__ == '__main__':
    plot_hota_slopegraph()
