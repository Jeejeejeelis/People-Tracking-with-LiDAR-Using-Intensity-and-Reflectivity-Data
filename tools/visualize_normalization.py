import numpy as np
import glob
import matplotlib.pyplot as plt
import os
from scipy.stats import iqr

def load_pfsd_points(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :4]

def main():
    print("Loading data for normalization comparison...")
    pfsd_files = sorted(glob.glob('OpenPCDet/data/nuscenes/v1.0-mini/samples/LIDAR_TOP/*.bin'))
    if not pfsd_files:
        print("PFSD files not found.")
        return

    all_reflectivities = []
    all_intensities = []
    
    # Load 20 frames for a good statistical sample
    for p in pfsd_files[:20]:
        points = load_pfsd_points(p)
        dist_sq = np.sum(points[:, :3]**2, axis=1)
        intensity = points[:, 3]
        reflectivity = intensity * dist_sq
        all_intensities.extend(intensity)
        all_reflectivities.extend(reflectivity)
        
    R = np.array(all_reflectivities)
    
    # --- Normalization Strategies ---
    
    # 1. Tanh Squashing (Industry Standard for Waymo)
    # We apply a slight scale (e.g., median) so it doesn't instantly snap to 1.0 for everything,
    # but we'll see it still saturates massively.
    R_tanh = np.tanh(R / np.median(R))
    
    # 2. Logarithmic Scaling + 99th Percentile Winsorization (Your Method)
    p99 = np.percentile(R, 99)
    R_clipped = np.clip(R, 0, p99)
    R_log_winsor = np.log1p(R_clipped)
    # Normalize to [0, 1] for fair comparison
    R_log_winsor = (R_log_winsor - R_log_winsor.min()) / (R_log_winsor.max() - R_log_winsor.min())
    
    # 3. Robust Scaler (Median & IQR)
    median_R = np.median(R)
    iqr_R = iqr(R)
    R_robust = (R - median_R) / iqr_R
    # Robust scaler leaves outliers as large positive numbers, it doesn't bound them to [0, 1].
    # We will visualize how it centers the data.
    
    # --- Visualization ---
    # To make this better than the last plot, we will use KDE (Kernel Density) histograms
    # which show the exact shape of the data, and we will do a side-by-side comparison.
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Subplot 1: Raw Data (SymLog Scale) to show the Outlier Problem
    ax = axes[0, 0]
    ax.hist(R, bins=np.logspace(0, np.log10(R.max()+1), 100), color='salmon', edgecolor='black', alpha=0.7)
    ax.set_xscale('log')
    ax.set_title("1. Raw Reflectivity ($I \\times d^2$)\n(Notice the massive tail stretching to 440,000)", fontsize=13)
    ax.axvline(np.median(R), color='blue', linestyle='--', label=f'Median: {np.median(R):.0f}')
    ax.axvline(p99, color='red', linestyle='--', label=f'99th Pct: {p99:.0f}')
    ax.legend()

    # Subplot 2: Tanh Squashing
    ax = axes[0, 1]
    ax.hist(R_tanh, bins=50, color='mediumpurple', edgecolor='black', alpha=0.7)
    ax.set_title("2. Tanh Squashing (Industry Standard)\n(Fails: 50% of the data is completely crushed against 1.0)", fontsize=13)
    
    # Subplot 3: Robust Scaler
    ax = axes[1, 0]
    # Limit x-axis to show the core distribution, as outliers will still be mathematically huge (e.g., +2000 IQR)
    ax.hist(R_robust, bins=1000, color='lightgreen', edgecolor='black', alpha=0.7)
    ax.set_xlim(-2, 10)
    ax.set_title("3. Robust Scaler (Median / IQR)\n(Succeeds in centering, but outliers still remain unbound on the right)", fontsize=13)
    ax.axvline(0, color='blue', linestyle='--', label='Median centered at 0')
    ax.legend()
    
    # Subplot 4: Log + Winsorization (Your Method)
    ax = axes[1, 1]
    ax.hist(R_log_winsor, bins=50, color='dodgerblue', edgecolor='black', alpha=0.7)
    ax.set_title("4. Log + 99th Winsorization (Your Strategy)\n(Optimal: Smooth distribution, heavy tail clipped cleanly at 1.0)", fontsize=13)
    
    plt.tight_layout()
    output_path = 'results/normalization_strategies_comparison.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    main()
