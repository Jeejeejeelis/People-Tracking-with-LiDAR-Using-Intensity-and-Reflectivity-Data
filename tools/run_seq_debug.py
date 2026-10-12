import cv2
import numpy as np
import os
from core.lidar_projection import load_velo, KittiProjection
from core.person_tracker import PersonTracker
from core.dl_tracker import OpenPCDetTracker

def run_sequence_tracking(seq_id="0017", num_frames=None, tracker_type="dbscan"):
    base_path = "data/training"
    
    # Map tracker_type to output folder and intensity mode
    out_suffix = tracker_type
    intensity_mode = 'none'
    
    if tracker_type == "dbscan":
        tracker = PersonTracker()
    elif tracker_type == "dl":
        intensity_mode = 'none'
        tracker = OpenPCDetTracker("OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml", "weights/pointpillar_7728.pth", intensity_mode=intensity_mode)
    elif tracker_type == "dl_intensity":
        intensity_mode = 'raw'
        tracker = OpenPCDetTracker("OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml", "weights/pointpillar_7728.pth", intensity_mode=intensity_mode)
    elif tracker_type == "dl_reflectivity":
        intensity_mode = 'compensated'
        out_suffix = "dl_intensity_reflectivity"
        tracker = OpenPCDetTracker("OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml", "weights/pointpillar_7728.pth", intensity_mode=intensity_mode)
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")

    out_dir = f"src/seq_debug_{seq_id}_{out_suffix}"
    os.makedirs(out_dir, exist_ok=True)

    kp = KittiProjection(os.path.join(base_path, f"calib/{seq_id}.txt"))
    
    if num_frames is None:
        num_frames = len(os.listdir(os.path.join(base_path, f"velodyne/{seq_id}")))
    
    for frame_id in range(num_frames):
        if frame_id % 10 == 0:
            print(f"Processing Seq {seq_id} ({tracker_type}) Frame {frame_id}/{num_frames}...")
        image_path = os.path.join(base_path, f"image_02/{seq_id}/{frame_id:06d}.png")
        velo_path = os.path.join(base_path, f"velodyne/{seq_id}/{frame_id:06d}.bin")
        
        img = cv2.imread(image_path)
        pc = load_velo(velo_path)
        
        if tracker_type == "dbscan":
            filtered_pc = tracker.preprocess(pc)
            clusters = tracker.cluster(filtered_pc)
            detections = tracker.filter_pedestrians(clusters)
        else:
            # DL tracker handles detection internally
            detections = tracker.run_detection(pc)
        
        # Track across frames
        tracked_peds = tracker.track(detections)
        
        # Draw our detections
        for ped in tracked_peds:
            p_min, p_max = ped['bbox']
            color = (0, 0, 255) if tracker_type == "dbscan" else (0, 255, 0)
            label = f"P{ped['id']}"
            
            corners = np.array([
                [p_min[0], p_min[1], p_min[2]], [p_max[0], p_min[1], p_min[2]],
                [p_max[0], p_max[1], p_min[2]], [p_min[0], p_max[1], p_min[2]],
                [p_min[0], p_min[1], p_max[2]], [p_max[0], p_min[1], p_max[2]],
                [p_max[0], p_max[1], p_max[2]], [p_min[0], p_max[1], p_max[2]]
            ])
            pts_2d, _ = kp.project_velo_to_image(corners)
            pts_2d = pts_2d.astype(int)
            
            if len(pts_2d) < 8: continue

            # Draw 3D Bounding Box in 2D
            for i in range(4):
                cv2.line(img, tuple(pts_2d[i]), tuple(pts_2d[(i+1)%4]), color, 2)
                cv2.line(img, tuple(pts_2d[i+4]), tuple(pts_2d[(i+1)%4+4]), color, 2)
                cv2.line(img, tuple(pts_2d[i]), tuple(pts_2d[i+4]), color, 2)
            
            # Project center for text label
            center_3d = ped['center'].reshape(1, 3)
            center_2d, _ = kp.project_velo_to_image(center_3d)
            if len(center_2d) > 0:
                cv2.putText(img, label, (int(center_2d[0][0]), int(center_2d[0][1])), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imwrite(os.path.join(out_dir, f"{frame_id:06d}.png"), img)
    print(f"Sequence {seq_id} ({tracker_type}) debug frames saved to {out_dir}")

if __name__ == "__main__":
    import sys
    seq = sys.argv[1] if len(sys.argv) > 1 else "0017"
    mode = sys.argv[2] if len(sys.argv) > 2 else "dbscan"
    frames = int(sys.argv[3]) if len(sys.argv) > 3 else None
    run_sequence_tracking(seq_id=seq, tracker_type=mode, num_frames=frames)
