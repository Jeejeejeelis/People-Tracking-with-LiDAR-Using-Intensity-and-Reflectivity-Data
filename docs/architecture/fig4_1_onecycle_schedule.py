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
    plt.rcParams.update({'font.family': 'serif'})
    
    epochs = np.linspace(0, 160, 500)
    
    lr = np.zeros_like(epochs)
    mom = np.zeros_like(epochs)
    
    # Phase 1: 0 to 48
    mask1 = epochs <= 48
    # Phase 2: 48 to 144
    mask2 = (epochs > 48) & (epochs <= 144)
    # Phase 3: 144 to 160
    mask3 = epochs > 144
    
    # LR calculations
    lr[mask1] = np.linspace(0.0003, 0.003, np.sum(mask1))
    # Cosine annealing
    progress = (epochs[mask2] - 48) / (144 - 48)
    lr[mask2] = 0.0003 + 0.5 * (0.003 - 0.0003) * (1 + np.cos(np.pi * progress))
    lr[mask3] = np.linspace(0.0003, 1e-5, np.sum(mask3))
    
    # Momentum calculations
    mom[mask1] = np.linspace(0.95, 0.85, np.sum(mask1))
    mom[mask2] = 0.95 - 0.5 * (0.95 - 0.85) * (1 + np.cos(np.pi * progress))
    mom[mask3] = np.linspace(0.95, 0.95, np.sum(mask3))
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color1 = '#1f77b4'
    ax1.set_xlabel('Training Epochs (0 - 160)')
    ax1.set_ylabel('Learning Rate $\eta$', color=color1)
    ax1.plot(epochs, lr, color=color1, linewidth=2, label='Learning Rate')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color2 = '#d62728'
    ax2.set_ylabel('AdamW Momentum $m$', color=color2)
    ax2.plot(epochs, mom, color=color2, linewidth=2, linestyle='--', label='Momentum')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add vertical lines for phases
    ax1.axvline(x=48, color='gray', linestyle=':', alpha=0.7)
    ax1.axvline(x=144, color='gray', linestyle=':', alpha=0.7)
    
    ax1.text(24, 1e-4, 'Phase 1\nWarmup', ha='center', va='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    ax1.text(96, 1e-4, 'Phase 2\nCosine Annealing', ha='center', va='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    ax1.text(152, 1e-4, 'Phase 3', ha='center', va='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    plt.title('OneCycle Super-Convergence Training Schedule')
    fig.tight_layout()
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(output_dir, 'fig4_1_onecycle_schedule.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig4_1_onecycle_schedule.pdf'))
    plt.close()

if __name__ == '__main__':
    main()
