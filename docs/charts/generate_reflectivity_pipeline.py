import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import mstats
import os

sns.set_theme(style="whitegrid")
out_dir = "/home/eelis/Desktop/Diplomityö/latex/First_Draft/thesisInserts"
os.makedirs(out_dir, exist_ok=True)

def generate_pipeline_visualization():
    # 1. Generate realistic synthetic LiDAR data
    np.random.seed(42)
    # Most distances are 5-40m, some outliers up to 80m
    distances = np.concatenate([np.random.normal(20, 10, 50000), np.random.normal(70, 5, 1000)])
    distances = np.clip(distances, 2.0, 100.0)
    
    # Intensity mostly 0-100, some bright retroreflectors at 255
    intensities = np.random.exponential(30, 51000)
    intensities = np.clip(intensities, 0, 255)
    
    # --- Step 1: Raw R_comp = I * d^2 ---
    raw_r = intensities * (distances ** 2)
    
    # --- Step 2: Log Transform ---
    # Add 1 to avoid log(0)
    log_r = np.log1p(raw_r)
    
    # --- Step 3: Winsorization (99th percentile) ---
    # We clip the top 1% of outliers
    p99 = np.percentile(log_r, 99)
    winsorized_r = np.clip(log_r, a_min=None, a_max=p99)
    
    # --- Step 4: Z-Score Standardization ---
    mean_r = np.mean(winsorized_r)
    std_r = np.std(winsorized_r)
    z_score_r = (winsorized_r - mean_r) / std_r
    
    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=300)
    
    # Colors for each step
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    
    # 1. Raw R_comp
    sns.histplot(raw_r, bins=50, ax=axes[0], color=colors[0], kde=True)
    axes[0].set_title(r'1. Raw Physics Equation ($R = I \cdot d^2$)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Reflectivity Magnitude', fontsize=11)
    axes[0].set_ylabel('Point Count', fontsize=11)
    # The raw data has a massive long tail, let's format the x-axis in scientific notation
    axes[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    axes[0].text(0.5, 0.7, 'Extreme Outliers\n(Long Tail)', transform=axes[0].transAxes, 
                 ha='center', fontsize=11, fontweight='bold', color='#333333', 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # 2. Log Transform
    sns.histplot(log_r, bins=50, ax=axes[1], color=colors[1], kde=True)
    axes[1].set_title(r'2. Logarithmic Compression ($\log$)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Log(Reflectivity)', fontsize=11)
    axes[1].set_ylabel('')
    axes[1].axvline(p99, color='red', linestyle='--', linewidth=2)
    axes[1].text(p99*0.95, axes[1].get_ylim()[1]*0.8, '99th Percentile', color='red', ha='right', fontweight='bold')
    
    # 3. Winsorization
    sns.histplot(winsorized_r, bins=50, ax=axes[2], color=colors[2], kde=True)
    axes[2].set_title(r'3. Robust Clipping (Winsorize 99%)', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Clipped Log(Reflectivity)', fontsize=11)
    axes[2].set_ylabel('')
    axes[2].text(0.5, 0.7, 'Outliers Collapsed\nto Upper Bound', transform=axes[2].transAxes, 
                 ha='center', fontsize=11, fontweight='bold', color='#333333', 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # 4. Z-Score Normalization
    sns.histplot(z_score_r, bins=50, ax=axes[3], color=colors[3], kde=True)
    axes[3].set_title(r'4. Standardization (Z-Score)', fontsize=13, fontweight='bold')
    axes[3].set_xlabel('Z-Score (Standard Deviations)', fontsize=11)
    axes[3].set_ylabel('')
    axes[3].axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
    axes[3].text(0, axes[3].get_ylim()[1]*0.9, 'Mean = 0', ha='center', va='bottom', fontweight='bold')
    
    # Add a main title to show the flow
    plt.suptitle(r'Radiometric Preprocessing Pipeline: $R_{comp} \rightarrow \log \rightarrow \text{Winsorize} \rightarrow \text{Z-score}$', 
                 fontsize=16, fontweight='bold', y=1.08)
    
    # Add arrows between subplots to emphasize the pipeline flow
    for i in range(3):
        # We draw an arrow in figure coordinates
        # Axes positions roughly: 0.125, 0.33, 0.53, 0.73 (width ~0.15)
        # We can just use annotate with xycoords='figure fraction'
        pass # Actually, text arrows might be cleaner. I'll rely on the titles "1. -> 2. -> 3. -> 4."
        
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'reflectivity_preprocessing_pipeline.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    
    # Save a PNG version too since they mentioned PNGs in that folder
    png_path = os.path.join(out_dir, 'reflectivity_preprocessing_pipeline.png')
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    
    print(f"Saved to {out_path} and {png_path}")

if __name__ == '__main__':
    generate_pipeline_visualization()
