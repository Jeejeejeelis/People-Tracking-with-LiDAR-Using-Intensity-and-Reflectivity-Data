import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from core.lidar_projection import KittiProjection, load_velo, visualize_projection

def run_visualization(seq_id="0017", frame_id=10):
    # Paths
    base_path = "/home/eelis/kitti/data/archive/kitti_tracking/testing"
    calib_path = os.path.join(base_path, f"calib/{seq_id}.txt")
    image_path = os.path.join(base_path, f"image_02/{seq_id}/{frame_id:06d}.png")
    velo_path = os.path.join(base_path, f"velodyne/{seq_id}/{frame_id:06d}.bin")
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # Load data
    img = cv2.imread(image_path)
    velo_points = load_velo(velo_path)
    
    # Initialize projection
    kp = KittiProjection(calib_path)
    
    # Project
    pts_2d, depths = kp.project_velo_to_image(velo_points)
    
    # Visualize
    overlay_img = visualize_projection(img.copy(), pts_2d, depths)
    
    # Save result
    output_path = f"/home/eelis/kitti/src/projection_{seq_id}_{frame_id:06d}.png"
    cv2.imwrite(output_path, overlay_img)
    print(f"Visualization saved to {output_path}")

    # Also show depth-colored version
    def visualize_depth_colored(img, pts_2d, depths, min_depth=1.0, max_depth=40.0):
        h, w, _ = img.shape
        mask = (depths > min_depth) & (depths < max_depth) & \
               (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) & \
               (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)
        
        pts_2d = pts_2d[mask]
        depths = depths[mask]
        
        # Color map
        norm_depths = (depths - min_depth) / (max_depth - min_depth)
        colors = plt.cm.jet(1 - norm_depths)[:, :3] * 255
        
        for i in range(len(pts_2d)):
            x, y = pts_2d[i].astype(int)
            cv2.circle(img, (x, y), 2, colors[i].tolist(), -1)
        return img

    depth_img = visualize_depth_colored(img.copy(), pts_2d, depths)
    depth_output_path = f"/home/eelis/kitti/src/projection_depth_{seq_id}_{frame_id:06d}.png"
    cv2.imwrite(depth_output_path, depth_img)
    print(f"Depth-colored visualization saved to {depth_output_path}")

if __name__ == "__main__":
    run_visualization()
