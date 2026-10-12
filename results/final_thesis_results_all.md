# Official Master's Thesis Results & Benchmark Tables

*Source: Master's Thesis Chapter 5 (`latex/Current_Version/chapters/05_results.tex`), Aalto University.*

---

## 1. 3D Pedestrian Detection Performance (3D mAP)

Evaluated across four Euclidean distance matching thresholds ($0.5$ m, $1.0$ m, $2.0$ m, and $4.0$ m) on the PFSD 128-beam validation set:

| Model Architecture | Feature Variant | mAP | AP @ 0.5m | AP @ 1.0m | AP @ 2.0m | AP @ 4.0m |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **PointPillars** | Baseline ($I$) | 45.70% | 21.77% | 33.96% | 53.39% | 73.70% |
| **PointPillars** | Reflectivity Swap ($R$) | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| **PointPillars** | Feature Decoupling ($I, d$) | 43.10% | 20.35% | 30.88% | 49.70% | 71.49% |
| **PointPillars** | Reflectivity Enhanced ($I, R$) | 0.45% | 0.00% | 0.00% | 0.00% | 1.79% |
| **PointPillars** | Multi-Modal ($I, R, d$) | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| **CenterPoint** | Baseline ($I$) | 76.65% | 71.61% | 74.48% | 78.60% | 81.92% |
| **CenterPoint** | Reflectivity Swap ($R$) | 39.03% | 34.94% | 36.67% | 40.54% | 43.98% |
| **CenterPoint** | Feature Decoupling ($I, d$) | 53.42% | 42.00% | 47.81% | 56.44% | 67.43% |
| **CenterPoint** | Reflectivity Enhanced ($I, R$) | 74.36% | 69.04% | 71.95% | 76.07% | 80.39% |
| **CenterPoint** | Multi-Modal ($I, R, d$) | **82.76%** | 77.40% | 80.76% | 84.65% | 88.21% |
| **VoxelNeXt** | Baseline ($I$) | 80.52% | 72.21% | 77.65% | 83.55% | 88.68% |
| **VoxelNeXt** | Reflectivity Swap ($R$) | 31.93% | 20.16% | 24.63% | 33.64% | 49.29% |
| **VoxelNeXt** | Feature Decoupling ($I, d$) | 79.85% | 71.48% | 76.93% | 82.86% | 88.12% |
| **VoxelNeXt** | Reflectivity Enhanced ($I, R$) | 79.20% | 73.27% | 76.30% | 81.48% | 85.74% |
| **VoxelNeXt** | Multi-Modal ($I, R, d$) | 78.35% | 72.28% | 75.50% | 80.66% | 84.95% |
| **HMFN** | Baseline ($I$) | **86.35%** | **80.76%** | **84.44%** | **88.10%** | **92.08%** |
| **HMFN** | Feature Decoupling ($I, d$) | 68.73% | 61.67% | 65.85% | 70.36% | 77.05% |
| **HMFN** | Reflectivity Enhanced ($I, R$) | 75.40% | 70.23% | 73.41% | 76.65% | 81.15% |
| **HMFN** | Multi-Modal ($I, R, d$) | 72.07% | 66.27% | 69.57% | 74.02% | 78.41% |
| **HMFN** | Reflectivity Swap ($R$) | 72.90% | 62.79% | 68.11% | 75.57% | 85.11% |

---

## 2. Multi-Model 3D Pedestrian Tracking Performance

Evaluated on PFSD validation sequences using the DeepSORT-lite Kalman filter tracker at optimal parameter settings:

| Model Architecture | Feature Variant | AMOTA ↑ | HOTA ↑ | IDF1 ↑ | False Positives (FP) ↓ |
|:---|:---|:---:|:---:|:---:|:---:|
| **CenterPoint** | Baseline ($I$) | **0.745** | 0.3973 | 0.3873 | 2,415 |
| **CenterPoint** | Reflectivity Swap ($R$) | 0.000 | 0.0000 | 0.0000 | 0 |
| **CenterPoint** | Feature Decoupling ($I, d$) | 0.727 | 0.3799 | 0.3651 | **540** |
| **CenterPoint** | Reflectivity Enhanced ($I, R$) | 0.716 | 0.3737 | 0.3602 | 2,786 |
| **CenterPoint** | Multi-Modal ($I, R, d$) | 0.681 | **0.4006** | **0.3912** | 682 |
| **VoxelNeXt** | Baseline ($I$) | 0.699 | 0.3175 | 0.3090 | 3,557 |
| **VoxelNeXt** | Reflectivity Swap ($R$) | 0.245 | 0.2013 | 0.1854 | 717 |
| **VoxelNeXt** | Feature Decoupling ($I, d$) | 0.701 | 0.3204 | 0.3154 | 641 |
| **VoxelNeXt** | Reflectivity Enhanced ($I, R$) | 0.703 | 0.3210 | 0.3167 | 3,675 |
| **VoxelNeXt** | Multi-Modal ($I, R, d$) | **0.711** | **0.3322** | **0.3241** | **534** |
| **HMFN** | Baseline ($I$) | 0.717 | 0.3659 | 0.3496 | 573 |
| **HMFN** | Reflectivity Swap ($R$) | 0.565 | 0.3146 | 0.3244 | 631 |
| **HMFN** | Feature Decoupling ($I, d$) | 0.657 | 0.3488 | 0.3407 | 3,544 |
| **HMFN** | Reflectivity Enhanced ($I, R$) | **0.719** | 0.3706 | 0.3720 | 598 |
| **HMFN** | Multi-Modal ($I, R, d$) | 0.697 | **0.3761** | **0.3727** | 616 |
| **PointPillars** | Baseline ($I$) | 0.000 | 0.1002 | 0.0893 | 719 |
| **PointPillars** | Reflectivity Swap ($R$) | 0.000 | 0.0777 | 0.0693 | 719 |
| **PointPillars** | Feature Decoupling ($I, d$) | 0.000 | 0.1782 | 0.1519 | 719 |
| **PointPillars** | Reflectivity Enhanced ($I, R$) | 0.000 | 0.0000 | 0.0000 | 719 |
| **PointPillars** | Multi-Modal ($I, R, d$) | 0.000 | 0.0000 | 0.0000 | 719 |

---

## 3. Optimal Tracker Parameters (Reproducibility)

The table records the exact distance gating threshold ($D$), confidence score cutoff ($F$), and reflectivity appearance penalty weight ($R$) identified via grid search:

| Model Architecture | Best AMOTA Parameters | Best HOTA Parameters | Best IDF1 Parameters |
|:---|:---|:---|:---|
| **CenterPoint Baseline** | (D: 40.0m, F: 0.6, R: 0.0x) | (D: 40.0m, F: 0.6, R: 0.0x) | (D: 40.0m, F: 0.6, R: 0.0x) |
| **CenterPoint Reflectivity Swap** | (D: 30.0m, F: 0.5, R: 0.0x) | (D: 30.0m, F: 0.5, R: 0.0x) | (D: 30.0m, F: 0.5, R: 0.0x) |
| **CenterPoint Feature Decoupling** | (D: 40.0m, F: 0.5, R: 0.0x) | (D: 30.0m, F: 0.6, R: 0.0x) | (D: 30.0m, F: 0.6, R: 0.0x) |
| **CenterPoint Reflectivity Enhanced** | (D: 30.0m, F: 0.5, R: 0.0x) | (D: 30.0m, F: 0.6, R: 0.0x) | (D: 30.0m, F: 0.6, R: 0.0x) |
| **CenterPoint Multi-Modal** | (D: 40.0m, F: 0.5, R: 0.0x) | (D: 40.0m, F: 0.5, R: 0.0x) | (D: 40.0m, F: 0.5, R: 0.0x) |
| **VoxelNeXt Baseline** | (D: 30.0m, F: 0.5, R: 0.0x) | (D: 40.0m, F: 0.5, R: 0.0x) | (D: 40.0m, F: 0.5, R: 0.0x) |
| **VoxelNeXt Reflectivity Swap** | (D: 30.0m, F: 0.5, R: 0.0x) | (D: 40.0m, F: 0.5, R: 0.16x) | (D: 40.0m, F: 0.5, R: 0.16x) |
| **VoxelNeXt Feature Decoupling** | (D: 30.0m, F: 0.5, R: 0.0x) | (D: 30.0m, F: 0.5, R: 0.0x) | (D: 30.0m, F: 0.5, R: 0.0x) |
| **VoxelNeXt Reflectivity Enhanced** | (D: 40.0m, F: 0.5, R: 0.0x) | (D: 30.0m, F: 0.5, R: 0.0x) | (D: 30.0m, F: 0.5, R: 0.0x) |
| **VoxelNeXt Multi-Modal** | (D: 30.0m, F: 0.5, R: 0.0x) | (D: 30.0m, F: 0.6, R: 0.0x) | (D: 40.0m, F: 0.5, R: 0.0x) |
| **HMFN Baseline** | (D: 40.0m, F: 0.5, R: 0.0x) | (D: 30.0m, F: 0.5, R: 0.0x) | (D: 30.0m, F: 0.5, R: 0.0x) |
| **HMFN Reflectivity Swap** | (D: 40.0m, F: 0.5, R: 0.22x) | (D: 40.0m, F: 0.5, R: 0.22x) | (D: 40.0m, F: 0.5, R: 0.22x) |
| **HMFN Feature Decoupling** | (D: 30.0m, F: 0.5, R: 0.0x) | (D: 30.0m, F: 0.6, R: 0.0x) | (D: 30.0m, F: 0.6, R: 0.0x) |
| **HMFN Reflectivity Enhanced** | (D: 40.0m, F: 0.5, R: 0.0x) | (D: 30.0m, F: 0.6, R: 0.0x) | (D: 40.0m, F: 0.6, R: 0.0x) |
| **HMFN Multi-Modal** | (D: 40.0m, F: 0.5, R: 0.0x) | (D: 40.0m, F: 0.6, R: 0.0x) | (D: 40.0m, F: 0.5, R: 0.0x) |
| **PointPillars Baseline** | (D: 40.0m, F: 0.5, R: 0.0x) | (D: 30.0m, F: 0.5, R: 0.0x) | (D: 30.0m, F: 0.5, R: 0.0x) |
| **PointPillars Reflectivity Swap** | (D: 30.0m, F: 0.5, R: 0.0x) | (D: 30.0m, F: 0.5, R: 0.0x) | (D: 30.0m, F: 0.5, R: 0.1x) |
| **PointPillars Feature Decoupling** | (D: 30.0m, F: 0.5, R: 0.0x) | (D: 30.0m, F: 0.6, R: 0.0x) | (D: 30.0m, F: 0.6, R: 0.0x) |
| **PointPillars Reflectivity Enhanced** | (D: 40.0m, F: 0.5, R: 0.0x) | (D: 30.0m, F: 0.6, R: 0.0x) | (D: 40.0m, F: 0.6, R: 0.0x) |
| **PointPillars Multi-Modal** | (D: 40.0m, F: 0.5, R: 0.0x) | (D: 40.0m, F: 0.6, R: 0.0x) | (D: 40.0m, F: 0.5, R: 0.0x) |

---

## 4. Experimental Affinity Reward Association

Evaluating material affinity attraction on the multi-modal networks ($I, R, d$):

| Architecture | Variant | Optimal Affinity Weight | AMOTA | HOTA | IDF1 |
|:---|:---|:---:|:---:|:---:|:---:|
| **CenterPoint** | Multi-Modal | 0.0 (Spatial Only) | 0.701 | 0.4000 | 0.3750 |
| **VoxelNeXt** | Multi-Modal | 0.0 (Spatial Only) | 0.739 | 0.3370 | 0.3270 |
| **HMFN** | Multi-Modal | **0.5 (Moderate Material Weighting)** | **0.717** | **0.3810** | **0.3700** |
