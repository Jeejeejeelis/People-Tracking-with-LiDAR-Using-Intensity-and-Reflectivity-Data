import matplotlib.pyplot as plt
import numpy as np
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
    
    milestones = [
        "1. Early Baseline (I)",
        "2. Fix #1: True 3D Voxel Backbone\n(+10-class, grid=0.05m)",
        "3. Fix #2: Training & Augmentation\n(gt_sampling, 160 epochs)",
        "4. Fix #3: Spatial Confidence Gating\n($w_{far} = 0.6$ cutoff >40m)"
    ]
    
    amota_scores = [0.0, 32.4, 58.1, 74.5]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(milestones))
    
    ax.plot(x, amota_scores, marker='o', markersize=10, color='#1f77b4', linewidth=3, zorder=3)
    
    for i, (score, desc) in enumerate(zip(amota_scores, milestones)):
        ax.annotate(f'{score}%\nAMOTA', (i, score), textcoords="offset points", xytext=(-20, 15), 
                    ha='right', fontsize=11, weight='bold', color='#d62728')
        
        y_offset = -15 if score > 50 else 15
        va_align = 'top' if score > 50 else 'bottom'
        
        ax.text(i, score + y_offset, desc, ha='center', va=va_align, 
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="gray", alpha=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels([f"Stage {i+1}" for i in range(len(milestones))])
    ax.set_ylabel('AMOTA (%)')
    ax.set_title('CenterPoint Optimization Trajectory', fontsize=14, weight='bold', pad=20)
    ax.set_ylim(-10, 100)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    ax.fill_between(x, amota_scores, color='#1f77b4', alpha=0.1)
    
    plt.tight_layout()
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(output_dir, 'fig6_1_centerpoint_trajectory.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig6_1_centerpoint_trajectory.pdf'))
    plt.close()

if __name__ == '__main__':
    main()
