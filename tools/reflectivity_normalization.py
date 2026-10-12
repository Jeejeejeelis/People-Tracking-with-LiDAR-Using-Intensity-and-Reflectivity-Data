import numpy as np
import os
import glob
import matplotlib.pyplot as plt

def load_kitti_points(path):
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return points

def load_pfsd_points(path):
    # PFSD in nuScenes format usually has 5 columns
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :4]
    return points

def calculate_reflectivity(points):
    dist_sq = np.sum(points[:, :3]**2, axis=1)
    # Ensure intensity is in 0-1 or 0-255? 
    # KITTI intensity is typically 0-1.
    # PFSD/nuScenes intensity can vary.
    intensity = points[:, 3]
    reflectivity = intensity * dist_sq
    return intensity, reflectivity

def analyze_dataset(paths, loader_func, name):
    all_intensities = []
    all_reflectivities = []
    
    # Sample 10 frames
    sample_paths = paths[:10]
    for p in sample_paths:
        points = loader_func(p)
        intensity, reflectivity = calculate_reflectivity(points)
        all_intensities.extend(intensity)
        all_reflectivities.extend(reflectivity)
        
    all_intensities = np.array(all_intensities)
    all_reflectivities = np.array(all_reflectivities)
    
    print(f"\n--- {name} Statistics ---")
    print(f"Intensity: mean={all_intensities.mean():.4f}, std={all_intensities.std():.4f}, min={all_intensities.min():.4f}, max={all_intensities.max():.4f}")
    print(f"Intensity Percentiles: 90th={np.percentile(all_intensities, 90):.4f}, 99th={np.percentile(all_intensities, 99):.4f}, 99.9th={np.percentile(all_intensities, 99.9):.4f}")
    print(f"Reflectivity (I*d^2): mean={all_reflectivities.mean():.4f}, std={all_reflectivities.std():.4f}, min={all_reflectivities.min():.4f}, max={all_reflectivities.max():.4f}")
    print(f"Reflectivity Percentiles: 90th={np.percentile(all_reflectivities, 90):.4f}, 99th={np.percentile(all_reflectivities, 99):.4f}, 99.9th={np.percentile(all_reflectivities, 99.9):.4f}")
    
    return all_intensities, all_reflectivities

def main():
    kitti_files = sorted(glob.glob('data/training/velodyne/0017/*.bin'))
    pfsd_files = sorted(glob.glob('OpenPCDet/data/nuscenes/v1.0-mini/samples/LIDAR_TOP/*.bin'))
    
    if not kitti_files:
        print("KITTI files not found.")
        return
    if not pfsd_files:
        print("PFSD files not found.")
        return
        
    ki, kr = analyze_dataset(kitti_files, load_kitti_points, "KITTI (Seq 0017)")
    pi, pr = analyze_dataset(pfsd_files, load_pfsd_points, "PFSD (mini)")
    
    # Plotting
    plt.figure(figsize=(12, 10))
    
    # Intensity Histogram
    plt.subplot(2, 2, 1)
    plt.hist(ki, bins=100, alpha=0.5, label='KITTI', density=True)
    plt.hist(pi, bins=100, alpha=0.5, label='PFSD', density=True)
    plt.title('Intensity Distribution (Normalized)')
    plt.legend()
    
    # Reflectivity Histogram
    plt.subplot(2, 2, 2)
    # Log scale for reflectivity as it can have large outliers
    plt.hist(np.log1p(kr), bins=100, alpha=0.5, label='KITTI', density=True)
    plt.hist(np.log1p(pr), bins=100, alpha=0.5, label='PFSD', density=True)
    plt.title('Log(1 + Reflectivity) Distribution')
    plt.legend()
    
    # Cumulative Intensity
    plt.subplot(2, 2, 3)
    plt.hist(ki, bins=100, alpha=0.5, label='KITTI', density=True, cumulative=True, histtype='step')
    plt.hist(pi, bins=100, alpha=0.5, label='PFSD', density=True, cumulative=True, histtype='step')
    plt.title('Cumulative Intensity')
    plt.legend()
    
    # Cumulative Reflectivity
    plt.subplot(2, 2, 4)
    plt.hist(np.log1p(kr), bins=100, alpha=0.5, label='KITTI', density=True, cumulative=True, histtype='step')
    plt.hist(np.log1p(pr), bins=100, alpha=0.5, label='PFSD', density=True, cumulative=True, histtype='step')
    plt.title('Cumulative Log(Reflectivity)')
    plt.legend()
    
    plt.tight_layout()
    output_path = 'results/reflectivity_comparison.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(output_path)
    print(f"\nHistogram saved to {output_path}")

if __name__ == "__main__":
    main()
