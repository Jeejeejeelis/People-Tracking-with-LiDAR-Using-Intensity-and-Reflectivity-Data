import numpy as np
import cv2
import os

class KittiProjection:
    def __init__(self, calib_path):
        self.calib = self.read_calib(calib_path)

    def read_calib(self, calib_path):
        """
        Reads calibration file and returns matrices.
        """
        calib_data = {}
        with open(calib_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                # Split by space and remove the trailing ':' from the first element
                parts = line.split()
                key = parts[0].replace(':', '')
                val = np.array([float(x) for x in parts[1:]])
                calib_data[key] = val
        
        # Reshape matrices based on actual keys in KITTI Tracking dataset
        # P2: 3x4 (Camera 2 projection matrix)
        P2 = calib_data['P2'].reshape(3, 4)
        
        # R_rect: 3x3 (Rectification matrix)
        # Note: In tracking dataset it's "R_rect", in raw it's "R0_rect"
        r_rect_key = 'R_rect' if 'R_rect' in calib_data else 'R0_rect'
        R_rect = calib_data[r_rect_key].reshape(3, 3)
        
        # Tr_velo_cam: 3x4 (Velodyne to Camera matrix)
        # Note: In tracking dataset it's "Tr_velo_cam", in raw it's "Tr_velo_to_cam"
        tr_velo_cam_key = 'Tr_velo_cam' if 'Tr_velo_cam' in calib_data else 'Tr_velo_to_cam'
        Tr_velo_cam = calib_data[tr_velo_cam_key].reshape(3, 4)
        
        return {
            'P2': P2,
            'R_rect': R_rect,
            'Tr_velo_cam': Tr_velo_cam
        }

    def project_velo_to_image(self, velo_points):
        """
        Projects Velodyne points to camera 2 image coordinates.
        """
        # 1. To camera coordinates: P_cam = Tr_velo_cam * P_velo
        n = velo_points.shape[0]
        velo_hom = np.hstack((velo_points[:, :3], np.ones((n, 1))))
        p_cam = np.dot(self.calib['Tr_velo_cam'], velo_hom.T).T
        
        # 2. To rectified camera coordinates: P_rect = R_rect * P_cam_hom
        # Note: R_rect is 3x3, P_cam is Nx3.
        p_rect = np.dot(self.calib['R_rect'], p_cam.T).T
        
        # 3. To image coordinates: P_img = P2 * [P_rect; 1]
        p_rect_hom = np.hstack((p_rect, np.ones((n, 1))))
        p_img = np.dot(self.calib['P2'], p_rect_hom.T).T
        
        # Normalize by last coord (z)
        depths = p_img[:, 2]
        p_img[:, 0] /= depths
        p_img[:, 1] /= depths
        
        return p_img[:, :2], depths

def load_velo(velo_path):
    """
    Loads Velodyne .bin file.
    """
    scan = np.fromfile(velo_path, dtype=np.float32)
    return scan.reshape((-1, 4))

def visualize_projection(img, pts_2d, depths, min_depth=1.0):
    """
    Overlays projected LiDAR points on image.
    """
    # Filter points in front of camera
    mask = depths > min_depth
    pts_2d = pts_2d[mask]
    depths = depths[mask]
    
    # Filter points within image boundaries
    h, w, _ = img.shape
    mask = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) & (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)
    pts_2d = pts_2d[mask]
    depths = depths[mask]
    
    # Draw points with color based on depth
    for i in range(len(pts_2d)):
        x, y = pts_2d[i].astype(int)
        # Simple color mapping based on depth (red is near, blue is far)
        color = (0, 255, 0) # Constant for now
        cv2.circle(img, (x, y), 1, color, -1)
    
    return img

if __name__ == "__main__":
    # This is a skeleton. Paths will need to be updated once dataset is downloaded.
    print("Kitti Projection script ready.")
