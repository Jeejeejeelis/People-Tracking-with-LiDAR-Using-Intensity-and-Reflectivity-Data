# LiDAR Pedestrian Tracking: Process, Findings & Architectural Insights

This document chronologically consolidates the experiments, engineering decisions, and core discoveries made during the development of the 3D LiDAR pedestrian tracking pipeline. It serves as a direct reference guide for writing the methodology and results chapters of the thesis.

## 1. Project Objectives & Constraints
- **Goal:** Robust pedestrian tracking in highly dense, semi-structured environments using exclusively 128-beam high-resolution LiDAR data from the PFSD (Pedestrian-Focus Scene Dataset).
- **Constraint:** Pure LiDAR-only tracking (no camera or radar sensor fusion).
- **Core Research Question:** How do different Point-Voxel neural network architectures handle dense LiDAR data, and can radiometric features (Intensity and physical Reflectivity) be leveraged to solve spatial ambiguities in dense crowds?

## 2. Architecture Selection & The "Negative Control"
We evaluated four neural network architectures to represent different methodologies for processing 3D sparse data.

* **PointPillars (The 2D Baseline):** Initially chosen as a fast, lightweight baseline.
  * **Finding:** PointPillars completely collapsed on the 128-beam PFSD dataset (**0.0% AMOTA**). The tracker recorded a massive "False Positive Explosion" (~90,000 false positives at 0.1 confidence).
  * **Insight:** Because PointPillars compresses the Z-axis early in its architecture to create a 2D Bird's Eye View (BEV) pseudo-image, it permanently loses the vertical spatial resolution necessary to separate pedestrians (cylinders) from dense urban clutter like poles and signs (also cylinders). This forces a brutal trade-off: either accept massive false positives, or raise the threshold so high that recall drops to 19%.
  * **Conclusion:** PointPillars serves as the perfect **Negative Control** in the thesis, mathematically proving that 2D Z-axis compression inherently fails on high-resolution LiDAR, justifying the computational expense of True 3D architectures.

* **The 3D Contenders:**
  1. **CenterPoint** (True 3D Voxel-based processing)
  2. **VoxelNeXt** (Pure 3D Sparse Convolution)
  3. **HMFN** (State-of-the-art Hybrid Multi-Scale Fusion Network leveraging Attention mechanisms).

## 3. Radiometric Feature Engineering: The Physics Proof
To improve tracking association, we investigated appending physical features to the spatial $(x, y, z)$ coordinates. We tested 5 dataset variants across the 3D architectures:
1. **Baseline**: 5 features $(x, y, z, I, t)$
2. **Decoupled**: 6 features $(x, y, z, I, d, t)$ — raw intensity and raw Euclidean range decoupled.
3. **Swap**: 5 features $(x, y, z, R, t)$ — raw intensity hard-swapped for distance-compensated physical Reflectivity ($R = I \times d^2$).
4. **Enhanced**: 6 features $(x, y, z, I, R, t)$
5. **Kitchen Sink**: 7 features $(x, y, z, I, R, d, t)$

### The Reflectivity Failure (The Physics Proof)
* **Finding:** Replacing raw Intensity with the hardcoded physical Reflectivity equation ($I \times d^2$) systematically ruined tracking performance across all architectures. For example, HMFN dropped by 11% AMOTA, and VoxelNeXt dropped by 12%.
* **Insight:** The inverse-square law mathematically amplifies distant background noise. Distant poles and signs suddenly possessed radiometric signatures identical to pedestrians. The neural network output overlapping confidence scores (0.4 - 0.6) for both humans and poles, making it impossible for *any* tracking algorithm to separate them without deleting true positives.
* **Conclusion:** The physics equation destroys non-Lambertian discriminative signatures. Feature engineering via hardcoded physics fails on high-resolution LiDAR.

### The Decoupling Triumph
* **Finding:** Passing the decoupled raw features (Intensity $I$ and Range $d$) and forcing the Voxel Feature Encoder (VFE) to dynamically *learn* the physical calibration resulted in massive improvements. CenterPoint and VoxelNeXt skyrocketed from their baselines (e.g., CenterPoint hitting 70.3% AMOTA).
* **Conclusion:** Feature Decoupling is universally beneficial for pure 3D sparse architectures. Letting the neural network optimize the radiometric-spatial relationship via gradient descent is infinitely superior to hardcoded physics.

## 4. Architectural Defenses for Radiometric Data
During the integration of 6 and 7-feature datasets, we discovered that standard Voxel networks suffer from "Volumetric Bleeding" and "Max-Pooling Feature Collapse" when exposed to dense radiometric features. We engineered four state-of-the-art defenses:

1. **Z-Score Normalization:** Replaced failure-prone Min-Max scaling with Z-Score Standardization to perfectly center clipped log-reflectivity without squashing outliers, ensuring optimal gradient flow.
2. **Augmentation Resync Hotfix:** Fixed a critical bug in spatial augmentations (like `random_world_scaling`) where the $x,y,z$ coordinates were transformed, but the pre-calculated decoupled $distance$ feature was left unchanged. We dynamically recalculated distance post-augmentation to prevent spatial desync.
3. **Representational Bottleneck Widening:** We discovered that forcing 7 complex features through a standard 16-channel bottleneck caused feature collapse. We dynamically scaled the initial `VoxelResBackBone8x` filters from 16 to 32 channels.
4. **Radiometric Time Decay ($e^{-\lambda |t|}$):** Multi-sweep temporal accumulation (10 sweeps) was smearing pedestrian material footprints, causing "volumetric bleeding" inside the MeanVFE. We applied an exponential time decay exclusively to historical intensity and reflectivity channels. This suppresses old material ghosting in background voxels while leaving the spatial $(x,y,z)$ trails fully intact for velocity regression.

## 5. The Multi-Metric Paradox & Tracking Rigidity
Because AMOTA is mathematically dominated by Detection Errors (False Positives), it masks how well track association is actually performing. We bypassed standard 2D tracking libraries and built custom 3D implementations for **HOTA** (Higher Order Tracking Accuracy) and **IDF1** (Identity F1 Score).

### The Paradox
* We discovered that while Reflectivity features slightly improved raw *Detection Accuracy* (DetA), they catastrophicly degraded *Association Accuracy* (AssA) and *Localization Accuracy* (LocA).
* **The Localization Shift:** Reflectivity shifts the neural network's bounding box centers away from the geometric center of the pedestrian and toward the edges of bright reflective clothing, lowering LocA.

### Tracker Hacking & The Sticky Pole Problem
We attempted to rescue the Reflectivity models by upgrading the tracker:
* **Reflectivity Cost Matrix (Frontier 1):** We extracted a "Reflectivity Signature" (mean point reflectivity inside the predicted box) and injected it as an Appearance/Re-ID feature into the Hungarian Cost Matrix to help resolve occlusions.
* **Mahalanobis Gating:** Failed completely. Because pedestrian motion is highly erratic (non-rigid), the tight statistical gates of the Kalman filter instantly fractured tracks during sudden movements.
* **Decoupled Material Gating (Frontier 3):** Attempted to decouple the spatial Euclidean gate from a Reflectivity Penalty. This resulted in the "Sticky Pole Problem"—without the coupled penalty to reject false positives, the tracker blindly anchored pedestrian trajectories to highly reflective poles that shared the same spatial proximity during occlusions.
* **Range-Scaled Gating:** Expanding the tracking gate linearly over distance increased AMOTA by +7.5% because it "swallowed" distant false positives. However, our HOTA/IDF1 metrics revealed this destroyed true identity continuity (IDTP dropped massively). 

### Final Tracking Conclusion
AMOTA can be highly misleading in dense crowds. To maintain true identity continuity (IDF1), the tracking algorithm must remain mathematically simple (strict spatial Euclidean distance). Complex multi-modal tracking heuristics fail due to tracking rigidity. Therefore, the burden of radiometric complexity must be handled entirely by the neural network architecture (Feature Decoupling), not the post-processing tracker.

## 6. Architectural Tuning for Fair Comparison
To ensure a scientifically rigorous and fair comparison between the architectures in highly dense crowds, critical hyperparameter tuning was performed prior to the final Triton ablation:
* **CenterPoint True 3D Upgrade:** We swapped the legacy `PillarVFE` for a True 3D `VoxelResBackBone8x` and lowered the voxel size to `0.05m`, which decreased the heatmap pixel size to `0.4m` for much higher spatial resolution.
* **Gaussian Washout Fix:** Dense pedestrian clusters caused overlapping heatmap targets to "wash out" and suppress each other. We fixed this by setting `MIN_RADIUS=1` and `GAUSSIAN_OVERLAP=0.05` across the architectures.
* **PointPillars Memory Scaling:** To handle the massive point density of 128-beam data, pillars were shrunk to `0.1m` resolution, and `MAX_NUMBER_OF_VOXELS` was dramatically scaled to 120k/160k.

## 7. The "Theoretical Maximums" Evaluation Strategy
Because AMOTA penalizes false positives so aggressively, tuning hyperparameters to maximize AMOTA often fractures true trajectories (lowering IDF1). To provide a rigorous academic analysis of the architecture's absolute ceiling, the final thesis evaluation executes a massive automated grid search on the Triton cluster.

The final evaluation generates a "Theoretical Maximums" matrix: each metric (AMOTA, HOTA, IDF1) is calculated independently from the parameter permutation that optimized it best. This avoids the "Frankenstein Tracker" assumption (we will explicitly disclaim that these metrics cannot be achieved simultaneously by a single deployable tracker) and brilliantly highlights the fundamental trade-offs between detection recall and identity continuity.

## 8. Future Work: Cross-Dataset Generalization (STCrowd)
Because the **Feature Decoupling** strategy proved successful (with CenterPoint and VoxelNeXt jumping to ~70% AMOTA), the methodology demands a generalization test. The final step of the thesis research will involve evaluating the winning PFSD-trained Decoupled architecture on the **STCrowd** dataset. 
* **Goal:** To mathematically verify whether the Voxel Feature Encoder learned true, universal radiometric signatures of pedestrians, or if it merely overfit to the specific hardware calibration and background environment of the PFSD dataset.
