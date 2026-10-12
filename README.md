# People Tracking with LiDAR Using Intensity and Reflectivity Data

> **Master's Thesis Project**  
> *A systematic study of radiometric LiDAR representations (Intensity vs. Distance-Compensated Reflectivity vs. Feature Decoupling) across 2D, 3D Sparse Voxel, and Hybrid Multi-Scale Attention architectures on dense 128-beam pedestrian crowds.*

---

## 📌 Abstract & Research Overview

Accurate 3D multi-object tracking (MOT) of pedestrians is vital for autonomous vehicles and mobile robots operating in crowded urban environments. While LiDAR spatial coordinates ($x, y, z$) provide precise geometry, raw return **intensity** ($I$) is severely attenuated by distance according to the radar range equation ($P_r \propto \frac{\sigma}{d^4}$ or $\frac{\sigma}{d^2}$ for extended Lambertian targets).

This thesis investigates whether augmenting deep neural networks and multi-object trackers with **distance-compensated reflectivity** ($R \approx I \cdot d^2$) or **feature decoupling** ($I, d$) enhances pedestrian detection and trajectory continuity in dense crowds.

Using the **PFSD (Pedestrian-Focus Scene Dataset)** featuring a high-density 128-beam LiDAR, we benchmark 4 distinct neural architectures across 4 input representations, evaluated via a **DeepSORT-lite Kalman Filter state machine** and the **Multi-Metric Triad** (AMOTA, MOTA, HOTA [DetA/AssA/LocA], and IDF1).

---

## 🔬 Core Discoveries & Academic Findings

1. **The 2D Compression Failure (PointPillars)**:  
   Architectures employing early vertical compression (collapsing the Z-axis into 2D pillars) inherently fail on dense 128-beam crowds, collapsing to **0.0% AMOTA**. They generate over 16,000 false-positive "ghost" detections because vertical noise (poles, street signs, foliage) cannot be separated from pedestrian torsos in bird's-eye view (BEV).

2. **The Reflectivity Paradox (The Physics Equation Failure)**:  
   Directly swapping raw Intensity with the hardcoded physical equation $R = I \cdot d^2$ catastrophically degrades tracking across all functioning architectures (HMFN drops from 71.7% to 36.8% AMOTA). Distance compensation artificially inflates distant retro-reflective background objects so heavily that their neural response distribution overlaps with pedestrians, causing tracking algorithms to fracture identities.

3. **The Feature Decoupling Triumph**:  
   Instead of hardcoding the physical calibration, feeding **decoupled intensity and range** ($I$ and $d$) allows 3D sparse voxel networks (CenterPoint, VoxelNeXt) to dynamically learn sensor-specific calibration curves natively. CenterPoint and VoxelNeXt both rocket from unviable baselines to **70.3% AMOTA**.

4. **HMFN Architectural Superiority**:  
   HMFN's native Hybrid Multi-Scale Attention handles dense feature extraction so effectively that its standard 5-feature baseline achieves the highest overall benchmark score (**71.7% AMOTA**).

5. **The Multi-Metric Triad Insight (Detection vs. Association)**:  
   Evaluating with HOTA reveals that while reflectivity signals enhance raw neural detection sensitivity (**DetA increases**), naive tracker matching suffers massive identity swaps (**AssA and IDF1 plummet**). Dynamic distance-dependent thresholding and calibrated reflectivity appearance signatures are required to recover association stability.

---

## 📊 Master Thesis Benchmark Results

All models were trained for **160 epochs** on the PFSD 128-beam LiDAR dataset and evaluated at a standardized `0.4` confidence threshold using the DeepSORT-lite Kalman tracker:

| Architecture | Input Representation | Input Channels | AMOTA ↑ | HOTA ↑ | DetA ↑ | AssA ↑ | LocA ↑ | IDF1 ↑ |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **HMFN** | **Baseline (Raw Intensity)** | $x, y, z, I, t$ | **71.7%** | 14.3% | 36.2% | 5.6% | **89.3%** | 10.1% |
| **HMFN** | Decoupled | $x, y, z, I, d, t$ | 68.9% | 13.4% | 33.8% | 5.3% | **89.3%** | 9.2% |
| **HMFN** | Kitchen Sink | $x, y, z, I, R, d, t$ | 68.0% | 13.9% | 35.9% | 5.4% | 88.2% | 9.8% |
| **HMFN** | Swap (Reflectivity) | $x, y, z, R, d, t$ | 36.8% | 28.9% | 46.2% | 18.1% | 84.0% | 28.1% |
| **CenterPoint** | True 3D Baseline | $x, y, z, I, t$ | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **CenterPoint** | **Decoupled** | $x, y, z, I, d, t$ | **70.3%** | 17.9% | 39.8% | 8.1% | 77.8% | 16.8% |
| **CenterPoint** | Kitchen Sink | $x, y, z, I, R, d, t$ | 68.2% | 25.3% | 52.1% | 12.3% | 87.0% | 22.7% |
| **CenterPoint** | Swap (Reflectivity) | $x, y, z, R, d, t$ | 68.6% | **38.6%** | **67.2%** | **22.2%** | 87.4% | **37.3%** |
| **VoxelNeXt** | Baseline | $x, y, z, I, t$ | 38.5% | 14.9% | 39.3% | 5.7% | 76.0% | 11.5% |
| **VoxelNeXt** | **Decoupled** | $x, y, z, I, d, t$ | **70.3%** | 21.7% | 43.4% | 10.8% | 88.0% | 18.7% |
| **VoxelNeXt** | Kitchen Sink | $x, y, z, I, R, d, t$ | 58.7% | 21.6% | 44.7% | 10.5% | 87.9% | 18.6% |
| **VoxelNeXt** | Swap (Reflectivity) | $x, y, z, R, d, t$ | 70.1% | 30.5% | 56.5% | 16.5% | 87.5% | 30.0% |
| **PointPillars** | Swap (Reflectivity) | $x, y, z, R, d, t$ | 0.0% | 17.9% | 30.3% | 10.6% | 69.0% | 15.1% |

---

## 🗂️ Repository Structure

```
├── core/                                # Core tracking and projection algorithms
│   ├── person_tracker.py                # DeepSORT-lite Kalman Filter state machine & Hungarian matching
│   ├── person_tracker_appearance.py     # Reflectivity signature appearance / Re-ID cost matrix tracker
│   ├── person_tracker_cascade.py        # Two-stage association cascade tracker
│   ├── person_tracker_reward.py         # Distance & reward-weighted tracker
│   ├── dl_tracker.py                    # Deep learning detector inference wrapper
│   ├── lidar_projection.py             # 3D LiDAR to 2D camera coordinate frame projection
│   └── token_to_scene.tsv               # PFSD token-to-scene lookup table
├── configs/                             # Model & dataset configurations
│   ├── pfsd_dataset_*.yaml              # Dataset definitions (baseline, decoupled, swap, kitchen_sink)
│   ├── pfsd_hmfn_*.yaml                 # HMFN variant configurations
│   ├── pfsd_centerpoint_*.yaml          # CenterPoint variant configurations
│   ├── pfsd_voxelnext_*.yaml            # VoxelNeXt variant configurations
│   └── pfsd_pp_multihead_*.yaml         # PointPillars variant configurations
├── PFSD/                                # Customized PFSD detection framework (forked OpenPCDet)
│   ├── pcdet/datasets/nuscenes/         # Feature-aware nuScenes/PFSD dataloader
│   └── ...
├── OpenPCDet/                           # OpenPCDet 3D perception framework
├── tools/                               # Evaluation, tuning, and visualization tools
│   ├── evaluate_pfsd_tracking.py        # Master PFSD tracking evaluator (AMOTA / MOTA)
│   ├── compute_pfsd_hota.py             # Computes HOTA, DetA, AssA, and LocA
│   ├── compute_pfsd_idf1.py             # Computes IDF1 trajectory continuity
│   ├── export_nusc_tracking_gt.py       # Ground-truth trajectory exporter
│   ├── extract_raw_predictions.py       # Caches unthresholded detections & reflectivity signatures
│   ├── tune_tracking_frontiers.py       # High-speed hyperparameter grid search for dynamic thresholds
│   ├── reflectivity_normalization.py   # Robust reflectivity filter (Winsorization + Z-score)
│   ├── run_pfsd_visual_debug.py         # 3D bounding box visual debug on camera feeds
│   ├── visualize_projection.py          # Point cloud to video feed projection
│   └── generate_thesis_tables.py        # Summary table generation
├── docs/                                # Thesis figures, charts, and methodology notes
│   ├── charts/                          # Publication figure generator scripts (.py) and PDFs
│   ├── architecture/                    # System architecture diagrams and schedules
│   ├── processAndFindings.md            # In-depth technical development log
│   └── tracker_justification.md         # Academic justification of Kalman filtering vs. IoU
├── results/                             # Benchmark evaluation logs and metrics summaries
│   ├── final_thesis_results_all.md      # Consolidated master metrics table
│   └── ...                              # Per-model JSON metrics and evaluation plots
├── weights/                             # Directory for model checkpoints (.pth)
│   └── README.md                        # Checkpoint inventory and instructions
├── requirements.txt                     # Python dependencies
├── setup_env.sh                         # Automated environment setup script
└── README.md
```

---

## ⚙️ Installation & Environment Setup

### Prerequisites
- Linux OS (tested on Ubuntu 20.04/22.04)
- NVIDIA GPU with CUDA 11.3+ or 12.x
- Python 3.8 – 3.10

### 1. Automated Setup
Run the setup script:
```bash
chmod +x setup_env.sh
./setup_env.sh
```

### 2. Manual Setup
If setting up manually:
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Compile PFSD / OpenPCDet CUDA extensions
cd PFSD
python setup.py develop
cd ..

# Set PYTHONPATH
export PYTHONPATH=$(pwd):$(pwd)/PFSD:$(pwd)/OpenPCDet:$PYTHONPATH
```

---

## 📦 Dataset Preparation

This project evaluates on the **Pedestrian-Focus Scene Dataset (PFSD)** in nuScenes format:

1. Download the PFSD dataset following the official [PFSD repository](https://github.com/VSlabRobotics/PFSD).
2. Organize your data directory as follows:
   ```
   data/pfsd/
   ├── v1.0-mini/
   │   ├── samples/
   │   │   ├── LIDAR_TOP/
   │   │   └── CAM_FRONT/
   │   ├── sweeps/
   │   ├── v1.0-mini/ (JSON metadata)
   │   └── pfsd_infos_10sweeps_val.pkl
   ```
3. Generate metadata pickle files using OpenPCDet tools:
   ```bash
   python -m pcdet.datasets.nuscenes.nuscenes_dataset \
       --func create_nuscenes_infos \
       --cfg_file configs/pfsd_dataset_baseline.yaml
   ```

---

## 🚀 Tracking Evaluation & Reproduction

### 1. Evaluate a Model Checkpoint
To evaluate 3D pedestrian tracking on PFSD using the DeepSORT-lite Kalman filter:
```bash
source venv/bin/activate
export PYTHONPATH=$(pwd):$(pwd)/PFSD:$(pwd)/OpenPCDet:$PYTHONPATH

python tools/evaluate_pfsd_tracking.py \
    --cfg_file configs/pfsd_hmfn_baseline.yaml \
    --ckpt weights/hmfn_baseline_160.pth \
    --score_thresh 0.4
```

### 2. Evaluate HOTA & Trajectory Continuity (IDF1)
Run the dedicated metric modules to compute HOTA (DetA, AssA, LocA) and IDF1:
```bash
# 1. Export ground truth trajectories
python tools/export_nusc_tracking_gt.py

# 2. Compute HOTA
python tools/compute_pfsd_hota.py --pred_dir results/hmfn_baseline_160/tracking_eval

# 3. Compute IDF1
python tools/compute_pfsd_idf1.py --pred_dir results/hmfn_baseline_160/tracking_eval
```

---

## 🎯 High-Speed Hyperparameter Tuning (Tracking Frontiers)

To tune dynamic distance-dependent thresholds and reflectivity penalty weights without re-running heavy neural network inference:

1. **Extract and cache unthresholded raw predictions**:
   ```bash
   python tools/extract_raw_predictions.py \
       --cfg_file configs/pfsd_centerpoint_swap.yaml \
       --ckpt weights/pfsd_centerpoint_swap_160.pth \
       --output_pkl results/cp_swap_raw.pkl
   ```

2. **Run grid search across millions of combinations in seconds**:
   ```bash
   python tools/tune_tracking_frontiers.py \
       --pkl results/cp_swap_raw.pkl \
       --dist_range 15 35 \
       --far_penalty 0.1 0.7
   ```

---

## 🖼️ Visualizations & Sensor Projection

### Project 3D Predictions onto Camera Video
```bash
python tools/run_pfsd_visual_debug.py \
    --cfg_file configs/pfsd_hmfn_baseline.yaml \
    --ckpt weights/hmfn_baseline_160.pth \
    --scene_name 0001
```

### Visualize Robust Reflectivity Normalization
```bash
python tools/visualize_normalization.py
```

---

## 📚 References & Acknowledgments

- **OpenPCDet**: OpenPCDet Development Team. [https://github.com/open-mmlab/OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
- **PFSD Dataset**: *PFSD: A Multi-Modal Pedestrian-Focus Scene Dataset for Rich Tasks in Semi-Structured Environments* (VSLab Robotics, 2025). [https://github.com/VSlabRobotics/PFSD](https://github.com/VSlabRobotics/PFSD)
- **NuScenes**: Caesar et al., *nuScenes: A multimodal dataset for autonomous driving*, CVPR 2020.
- **TrackEval**: Luiten et al., *TrackEval: Multi-Object Tracking Evaluation Toolkit*, 2020.
