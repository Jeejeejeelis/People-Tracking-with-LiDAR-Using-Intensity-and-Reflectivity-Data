import cv2
import numpy as np
import os

def create_final_comparison_grid():
    out_dir = "visualizations/combined/finalComparison"
    os.makedirs(out_dir, exist_ok=True)
    
    empty_dir = "visualizations/empty"
    vn_dir = "visualizations/voxelnext/baseline"
    cp_dir = "visualizations/centerpoint/MultiModal(I, R, d)"
    hmfn_dir = "visualizations/hmfn/MultiModal(I, R, d)"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 3
    color = (255, 255, 255)
    
    def add_title(img, title):
        if img is None:
            return None
        # Add black banner at the top
        banner_h = 60
        h, w, c = img.shape
        out = np.zeros((h + banner_h, w, c), dtype=np.uint8)
        out[banner_h:, :, :] = img
        # Add text
        text_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = int(banner_h / 2 + text_size[1] / 2)
        cv2.putText(out, title, (text_x, text_y), font, font_scale, color, font_thickness)
        return out

    num_frames = len([f for f in os.listdir(empty_dir) if f.endswith('.jpg')])
    print(f"Combining {num_frames} frames for FINAL COMPARISON")
    
    for i in range(num_frames):
        filename = f"{i:03d}.jpg"
        
        # Load images
        img_vn = cv2.imread(os.path.join(vn_dir, filename))
        img_cp = cv2.imread(os.path.join(cp_dir, filename))
        img_hmfn = cv2.imread(os.path.join(hmfn_dir, filename))
        img_empty = cv2.imread(os.path.join(empty_dir, filename))
        
        # Blank placeholder
        if img_empty is not None:
            h, w, c = img_empty.shape
        elif img_vn is not None:
            h, w, c = img_vn.shape
        else:
            h, w, c = 900, 1600, 3 # fallback
        blank = np.zeros((h, w, c), dtype=np.uint8)
        
        if img_vn is None: img_vn = blank.copy()
        if img_cp is None: img_cp = blank.copy()
        if img_hmfn is None: img_hmfn = blank.copy()
        if img_empty is None: img_empty = blank.copy()
        
        # Add titles
        top_left = add_title(img_vn, "VoxelNeXt Baseline (Lowest IDF1)")
        top_right = add_title(img_cp, "CenterPoint Multi-Modal (Highest IDF1)")
        bottom_left = add_title(img_hmfn, "HMFN Multi-Modal (Dataset Model)")
        bottom_right = add_title(img_empty, "Clean View (RGB)")
        
        # Concatenate
        top_row = np.hstack((top_left, top_right))
        bottom_row = np.hstack((bottom_left, bottom_right))
        combined = np.vstack((top_row, bottom_row))
        
        # Save
        out_path = os.path.join(out_dir, filename)
        cv2.imwrite(out_path, combined)
        
        if i % 10 == 0:
            print(f"Processed frame {i}/{num_frames}")
            
    print(f"Combined frames saved to {out_dir}!")

if __name__ == "__main__":
    create_final_comparison_grid()
