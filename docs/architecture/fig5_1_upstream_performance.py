import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    try:
        plt.style.use('seaborn-v0_8-paper')
    except:
        try:
            plt.style.use('seaborn-paper')
        except:
            pass
    plt.rcParams.update({'font.family': 'sans-serif'})
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: Upstream 3D mAP mock or provided data
    labels_A = ['0.5m', '1.0m', '2.0m', '4.0m']
    x = np.arange(len(labels_A))
    width = 0.2
    
    # Mock data to illustrate trends based on architecture complexity
    pp_map = [20.1, 25.4, 30.2, 35.1]
    cp_map = [40.5, 50.2, 58.4, 65.2]
    vn_map = [45.2, 55.1, 62.3, 70.4]
    hmfn_map = [55.4, 65.3, 75.1, 82.5]
    
    ax1.bar(x - 1.5*width, pp_map, width, label='PointPillars', color='#1f77b4')
    ax1.bar(x - 0.5*width, cp_map, width, label='CenterPoint', color='#ff7f0e')
    ax1.bar(x + 0.5*width, vn_map, width, label='VoxelNeXt', color='#2ca02c')
    ax1.bar(x + 1.5*width, hmfn_map, width, label='HMFN', color='#d62728')
    
    ax1.set_ylabel('3D mAP (%)')
    ax1.set_xlabel('Distance Thresholds')
    ax1.set_title('Panel A: Upstream 3D mAP Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_A)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Panel B: Physical Reflectivity Collapse
    models = ['PointPillars\nSwap (R, d)', 'VoxelNeXt\nSwap (R, d)', 'VoxelNeXt\nDecoupled (I, d)', 'HMFN\nMulti-Modal']
    map_scores = [0.0, 31.93, 54.2, 71.9] 
    
    colors = ['#1f77b4', '#2ca02c', '#2ca02c', '#d62728']
    
    bars = ax2.bar(models, map_scores, color=colors, alpha=0.8)
    ax2.set_ylabel('Score (%)')
    ax2.set_title('Panel B: Physical Reflectivity Resilience')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 80)
    
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), 
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, weight='bold')
                     
    plt.tight_layout()
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(output_dir, 'fig5_1_upstream_performance.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig5_1_upstream_performance.pdf'))
    plt.close()

if __name__ == '__main__':
    main()
