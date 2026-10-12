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
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times']})
    
    t = np.linspace(0, 1.0, 100)
    w = np.exp(-2.5 * t)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, w, color='#d62728', linewidth=2.5, label=r'$w(t) = e^{-2.5t}$')
    
    # Key points
    key_t = [0.0, 0.1, 0.5, 1.0]
    key_w = np.exp(-2.5 * np.array(key_t))
    labels = [
        "Sweep 0\n(Current)\n$w=1.000$",
        "Sweep -1\n$w=0.779$",
        "Sweep -5\n$w=0.287$",
        "Sweep -10\n$w=0.082$\n(92% Suppressed)"
    ]
    
    ax.scatter(key_t, key_w, color='#1f77b4', s=60, zorder=5)
    
    for i, (kt, kw, lbl) in enumerate(zip(key_t, key_w, labels)):
        xytext_offset = (10, 10) if i < 2 else (10, 15)
        if kt == 1.0:
            xytext_offset = (-10, 20)
            ha = 'right'
        else:
            ha = 'left'
        
        ax.annotate(lbl, (kt, kw), textcoords="offset points", xytext=xytext_offset,
                    ha=ha, fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="gray"))
    
    ax.set_xlabel('Time Delay $t$ (seconds)')
    ax.set_ylabel('Influence Weight $w(t)$')
    ax.set_title('Exponential Anti-Ghosting Temporal Decay')
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(output_dir, 'fig3_3_anti_ghosting_decay.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig3_3_anti_ghosting_decay.pdf'))
    plt.close()

if __name__ == '__main__':
    main()
