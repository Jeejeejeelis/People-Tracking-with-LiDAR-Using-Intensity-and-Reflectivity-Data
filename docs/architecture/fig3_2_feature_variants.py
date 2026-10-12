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
    plt.rcParams.update({'font.family': 'sans-serif'})
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    variants = [
        {"name": "Baseline (I)\n5 Channels", "channels": [
            ("x", "blue"), ("y", "blue"), ("z", "blue"), ("I", "orange"), ("$\Delta t$", "purple")
        ]},
        {"name": "Decoupled (I, d)\n6 Channels", "channels": [
            ("x", "blue"), ("y", "blue"), ("z", "blue"), ("I", "orange"), ("d", "teal"), ("$\Delta t$", "purple")
        ]},
        {"name": "Swap (R, d)\n6 Channels", "channels": [
            ("x", "blue"), ("y", "blue"), ("z", "blue"), ("$R_{comp}$", "orange"), ("d", "teal"), ("$\Delta t$", "purple")
        ]},
        {"name": "Multi-Modal ($I, R_{log}, d$)\n7 Channels", "channels": [
            ("x", "blue"), ("y", "blue"), ("z", "blue"), ("I", "orange"), ("$\hat{R}_{log}$", "orange"), ("d", "teal"), ("$\Delta t$", "purple")
        ]}
    ]
    
    colors_map = {
        "blue": "#1f77b4",
        "orange": "#ff7f0e",
        "teal": "#008080",
        "purple": "#9467bd"
    }
    
    y_offset = 4.8
    
    for i, var in enumerate(variants):
        ax.text(0.2, y_offset + 0.3, var["name"], va='center', ha='left', fontsize=11, weight='bold')
        
        x_offset = 2.8
        for ch, col in var["channels"]:
            rect = patches.Rectangle((x_offset, y_offset), 0.8, 0.6, 
                                     edgecolor='black', facecolor=colors_map[col], alpha=0.6, lw=1.5)
            ax.add_patch(rect)
            ax.text(x_offset + 0.4, y_offset + 0.3, ch, va='center', ha='center', fontsize=10, weight='bold')
            x_offset += 0.85
            
        y_offset -= 1.3
        
    legend_elements = [
        patches.Patch(facecolor=colors_map["blue"], alpha=0.6, edgecolor='black', label='Spatial $[x,y,z]$'),
        patches.Patch(facecolor=colors_map["orange"], alpha=0.6, edgecolor='black', label='Radiometric $[I, R, \\hat{R}_{log}]$'),
        patches.Patch(facecolor=colors_map["teal"], alpha=0.6, edgecolor='black', label='Range $[d]$'),
        patches.Patch(facecolor=colors_map["purple"], alpha=0.6, edgecolor='black', label='Temporal $[\\Delta t]$')
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05), frameon=False)
    
    plt.title('Input Point Feature Variants', fontsize=14, weight='bold')
    plt.tight_layout()
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(output_dir, 'fig3_2_feature_variants.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig3_2_feature_variants.pdf'))
    plt.close()

if __name__ == '__main__':
    main()
