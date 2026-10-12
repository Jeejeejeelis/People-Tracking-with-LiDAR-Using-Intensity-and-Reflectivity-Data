import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def main():
    try:
        plt.style.use('seaborn-v0_8-paper')
    except:
        try:
            plt.style.use('seaborn-paper')
        except:
            pass
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif']})
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    colors = {
        'data': '#1f77b4',       # Blue
        'preprocess': '#ff7f0e', # Orange
        'detector': '#2ca02c',   # Green
        'tracking': '#9467bd'    # Purple
    }
    
    blocks = [
        {"x": 0.5, "w": 2.2, "color": colors['data'], 
         "text": "1. Input Data\n\n10-Sweep Accumulation\n& SE(3) Alignment\n($\Delta t = 1.0$ s)"},
        {"x": 3.3, "w": 2.6, "color": colors['preprocess'], 
         "text": "2. Radiometric Preproc\n\n$I, d \\rightarrow R_{comp} \\rightarrow R_{log}$\nWinsorization $\\rightarrow$ Z-Score\n$\\hat{R}_{log}$"},
        {"x": 6.5, "w": 2.2, "color": colors['detector'], 
         "text": "3. 3D Detector\n\nVoxelization (0.05m)\n$\downarrow$\n3D VFE $\\rightarrow$ Backbones\n(VoxelNeXt/CenterPoint/HMFN)"},
        {"x": 9.3, "w": 2.5, "color": colors['tracking'], 
         "text": "4. 3D DeepSORT-lite\n\n6D Kalman Filter\nAnti-Ghosting $w(t)=e^{-2.5t}$\n3D Hungarian Matching"}
    ]
    
    y = 1.0
    h = 2.0
    
    for i, b in enumerate(blocks):
        rect = patches.FancyBboxPatch(
            (b["x"], y), b["w"], h,
            boxstyle="round,pad=0.1",
            edgecolor='black',
            facecolor=b["color"],
            alpha=0.2,
            linewidth=1.5
        )
        ax.add_patch(rect)
        
        # Border patch for stronger edge
        rect_edge = patches.FancyBboxPatch(
            (b["x"], y), b["w"], h,
            boxstyle="round,pad=0.1",
            edgecolor=b["color"],
            facecolor='none',
            linewidth=2
        )
        ax.add_patch(rect_edge)
        
        ax.text(b["x"] + b["w"]/2, y + h/2, b["text"], 
                ha='center', va='center', fontsize=10, 
                color='black', weight='bold')
        
        # Draw arrows between blocks
        if i < len(blocks) - 1:
            ax.annotate('', xy=(blocks[i+1]["x"], y + h/2), xytext=(b["x"] + b["w"], y + h/2),
                        arrowprops=dict(arrowstyle="->", lw=2, color='gray'))

    plt.title('End-to-End Perception and Tracking Pipeline', fontsize=14, pad=20, weight='bold')
    plt.tight_layout()
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(output_dir, 'fig3_1_system_architecture_diagram.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig3_1_system_architecture_diagram.pdf'))
    plt.close()

if __name__ == '__main__':
    main()
