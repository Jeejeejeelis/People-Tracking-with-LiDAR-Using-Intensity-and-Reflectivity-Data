import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

sns.set_theme(style="whitegrid")
out_dir = "/home/eelis/Desktop/Diplomityö/peopleTrackingWithLidar/docs_and_notes/charts"
csv_file = "/home/eelis/Desktop/Diplomityö/peopleTrackingWithLidar/docs_and_notes/charts/full_ablation_grid.csv"

def plot_fitted_curves():
    df = pd.read_csv(csv_file)
    
    # Target models designed for reflectivity
    target_models = ['vn_swap', 'hmfn_swap']
    metric = 'IDF1_Pct'
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    titles = ['VoxelNeXt Swap (Forced Physics)', 'HMFN Swap (Forced Physics)']
    colors = ['#1f77b4', '#d62728']
    
    for i, model in enumerate(target_models):
        ax = axes[i]
        
        # Get mean data for this model
        model_df = df[df['Architecture_Variant'] == model]
        grouped = model_df.groupby('Reflectivity_Penalty_Weight')[metric].mean().reset_index()
        
        x = grouped['Reflectivity_Penalty_Weight'].values
        y = grouped[metric].values
        
        # Plot discrete sampled points
        ax.scatter(x, y, color=colors[i], s=150, zorder=5, label='Sampled Grid Points', edgecolor='black', lw=1.5)
        
        # Fit a 2nd-degree polynomial (quadratic curve) to find the theoretical peak
        # y = ax^2 + bx + c
        p = np.polyfit(x, y, 2)
        poly = np.poly1d(p)
        
        # Generate continuous X values for the smooth curve
        x_smooth = np.linspace(0, 0.5, 200)
        y_smooth = poly(x_smooth)
        
        # Plot the fitted curve
        ax.plot(x_smooth, y_smooth, color=colors[i], linestyle='--', lw=2.5, alpha=0.7, label='Fitted Polynomial Curve')
        
        # Find the theoretical optimum (vertex of parabola: x = -b / 2a)
        a, b, c = p
        if a < 0: # Ensures it's a peak, not a valley
            opt_x = -b / (2 * a)
            if 0 <= opt_x <= 0.6: # If peak is within reasonable bounds
                opt_y = poly(opt_x)
                ax.scatter([opt_x], [opt_y], color='gold', s=250, marker='*', zorder=10, edgecolor='black', lw=1.5, label=f'Theoretical Optimum\n(Refl={opt_x:.3f}x)')
                
                # Draw lines pointing to the optimum
                ax.axvline(x=opt_x, color='gray', linestyle=':', ymin=0, ymax=(opt_y - ax.get_ylim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0]))
                ax.annotate(f"Peak @ {opt_x:.2f}x", (opt_x, opt_y), xytext=(15, -15), 
                            textcoords='offset points', fontsize=11, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.9))
        
        ax.set_title(titles[i], fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Reflectivity Penalty Weight (x)', fontsize=12)
        ax.set_ylabel('Mean IDF1 (%)', fontsize=12, fontweight='bold')
        ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        
        # Slight y-axis expansion for aesthetics
        y_range = max(y_smooth) - min(y)
        ax.set_ylim(min(y) - y_range*0.5, max(y_smooth) + y_range*0.8)
        
        ax.legend(loc='lower right', fontsize=10, frameon=True)
    
    plt.suptitle('Polynomial Curve Fitting for Tracking Optimums\n(Interpolating the True Mathematical Peak of Reflectivity Models)', fontsize=16, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '12_fitted_optimum_curves.pdf'), bbox_inches='tight')

if __name__ == '__main__':
    plot_fitted_curves()
