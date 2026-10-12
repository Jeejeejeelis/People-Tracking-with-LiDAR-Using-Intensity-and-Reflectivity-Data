# Pre-trained Model Weights

This directory is intended to store the pre-trained and fine-tuned model checkpoints (`.pth` files) for the 4 evaluated 3D detection architectures across their feature representation variants.

Due to file size constraints, large `.pth` model weights are excluded from Git version control (as configured in `.gitignore`).

---

## Model Checkpoint Inventory

All models were trained for 160 epochs on the **PFSD (Pedestrian-Focus Scene Dataset)** 128-beam LiDAR dataset:

| Architecture | Variant | Input Features | Checkpoint Filename | Size |
|:---|:---|:---|:---|:---|
| **HMFN** | Baseline | 5-feat ($x, y, z, I, t$) | `hmfn_baseline_160.pth` | ~272 MB |
| **HMFN** | Decoupled | 6-feat ($x, y, z, I, d, t$) | `hmfn_decoupled_160.pth` | ~259 MB |
| **HMFN** | Swap (Reflectivity) | 6-feat ($x, y, z, R, d, t$) | `hmfn_swap_epoch_160.pth` | ~259 MB |
| **HMFN** | Kitchen Sink | 7-feat ($x, y, z, I, R, d, t$) | `pfsd_hmfn_kitchen_sink_160.pth` | ~259 MB |
| **HMFN** | Enhanced Tuned | 6-feat ($x, y, z, I, R, t$) | `hmfn_enhanced_tuned_160.pth` | ~259 MB |
| **CenterPoint** | True 3D Baseline | 5-feat ($x, y, z, I, t$) | `cp_baseline_tuned_160.pth` | ~106 MB |
| **CenterPoint** | Decoupled | 6-feat ($x, y, z, I, d, t$) | `cp_decoupled_160.pth` | ~107 MB |
| **CenterPoint** | Swap (Reflectivity) | 6-feat ($x, y, z, R, d, t$) | `pfsd_centerpoint_swap_160.pth` | ~106 MB |
| **CenterPoint** | Kitchen Sink | 7-feat ($x, y, z, I, R, d, t$) | `pfsd_centerpoint_kitchen_sink.pth` | ~107 MB |
| **CenterPoint** | Enhanced Tuned | 6-feat ($x, y, z, I, R, t$) | `cp_enhanced_tuned_160.pth` | ~107 MB |
| **VoxelNeXt** | Baseline | 5-feat ($x, y, z, I, t$) | `voxelnext_baseline_160.pth` | ~92 MB |
| **VoxelNeXt** | Decoupled | 6-feat ($x, y, z, I, d, t$) | `vn_decoupled_160.pth` | ~92 MB |
| **VoxelNeXt** | Swap (Reflectivity) | 6-feat ($x, y, z, R, d, t$) | `pfsd_voxelnext_swap_160.pth` | ~92 MB |
| **VoxelNeXt** | Kitchen Sink | 7-feat ($x, y, z, I, R, d, t$) | `pfsd_voxelnext_kitchen_sink_160.pth` | ~92 MB |
| **VoxelNeXt** | Enhanced Tuned | 6-feat ($x, y, z, I, R, t$) | `vn_enhanced_tuned_160.pth` | ~92 MB |
| **PointPillars** | Baseline | 5-feat ($x, y, z, I, t$) | `pp_baseline_160.pth` | ~70 MB |
| **PointPillars** | Decoupled | 6-feat ($x, y, z, I, d, t$) | `pp_decoupled_160.pth` | ~70 MB |
| **PointPillars** | Swap (Reflectivity) | 6-feat ($x, y, z, R, d, t$) | `pp_swap_160.pth` | ~70 MB |
| **PointPillars** | Kitchen Sink | 7-feat ($x, y, z, I, R, d, t$) | `pp_enhanced_160.pth` | ~70 MB |

---

## Instructions for Loading Weights

Place downloaded `.pth` weights into this `weights/` directory.

To run evaluation using a specific checkpoint:
```bash
python tools/evaluate_pfsd_tracking.py \
    --cfg_file configs/pfsd_hmfn_baseline.yaml \
    --ckpt weights/hmfn_baseline_160.pth \
    --score_thresh 0.4
```
