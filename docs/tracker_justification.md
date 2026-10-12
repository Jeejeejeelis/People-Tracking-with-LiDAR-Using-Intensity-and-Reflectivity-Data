# Architectural Justification: Upstream Diversity vs. Downstream Rigidity

## 1. Introduction
In standard 3D Multi-Object Tracking (MOT) pipelines, the system is fundamentally bipartite: an **Upstream Detector** (which extracts spatial features and categorizes point clouds into bounding boxes) and a **Downstream Tracker** (which associates those boxes across time into continuous identities). 

For the purposes of this thesis—which evaluates pedestrian tracking in dense 128-beam LiDAR crowds (PFSD)—our methodology intentionally employs **architectural diversity upstream** (evaluating PointPillars, CenterPoint, VoxelNeXt, and HMFN) while rigidly enforcing a **single, unified tracking heuristic downstream** (a pure spatial Euclidean Kalman Filter). 

This document justifies this methodological design, explaining why hand-crafted radiometric trackers fail and why material feature extraction must occur inside the deep learning Voxel Feature Encoder (VFE).

---

## 2. The Upstream Ablation: Why Multiple Detection Architectures?
Evaluating multiple detection architectures is mandatory because different point-cloud encoding strategies react fundamentally differently to dense crowds and distance-compensated reflectivity ($I \times d^2$):

1. **PointPillars (2D Z-Axis Compression):** Acts as a negative control. By crushing the Z-axis into 2D pseudo-images early in the network, it blurs pedestrians into vertical noise (e.g., street signs, poles), proving that 2D architectures cannot handle dense 128-beam data.
2. **CenterPoint & VoxelNeXt (3D Sparse Convolutions):** These architectures preserve true 3D topology. Testing them with and without the "Decoupled" reflectivity features allows us to measure exactly how deep 3D networks dynamically learn physical material calibrations.
3. **HMFN (Hybrid Multi-Scale Fusion):** Represents the state-of-the-art baseline, designed specifically to fuse multi-scale features in semi-structured environments.

By testing these models against each other, the thesis isolates how effectively the neural network's architecture can leverage complex spatial-radiometric signatures before any tracking logic is applied.

---

## 3. The Downstream Standardization: Why a Single Euclidean Tracker?
To scientifically evaluate the upstream models, the downstream tracker must remain a strict, unvarying constant. Introducing complex tracking heuristics (like material-based Re-Identification) introduces confounding variables—making it impossible to determine if a jump in AMOTA is due to a superior 3D detector or a hyper-tuned tracker.

Furthermore, extensive empirical testing during this research revealed that **hand-crafted downstream material tracking mathematically degrades identity preservation**. We experimented with two advanced radiometric tracking paradigms:

### A. The Affinity-Reward Tracker (Maximum-Weight Bipartite Matching)
We attempted to augment the spatial Kalman filter with an additive reward based on the Bhattacharyya similarity of 10-bin reflectivity histograms. 
* **Result:** While frame-by-frame recall (AMOTA) rose slightly, long-term identity continuity (HOTA and IDF1) collapsed.

### B. The Two-Stage Cascade (DeepSORT-Style)
We decoupled kinematics from materials entirely, using pure spatial matching for active tracks (Stage 1) and pure reflectivity matching for occluded/lost tracks (Stage 2).
* **Result:** Performance worsened across all metrics.

### 4. The Theoretical Failure of Downstream Material Heuristics
The failure of these advanced tracking paradigms stems from two physical constraints of 128-beam LiDAR:

1. **Physical Anisotropy ($\cos\theta$) and Articulation:**
   The physical LiDAR equation dictates that returned intensity depends heavily on the local incidence angle ($\cos\theta$). Pedestrians are articulated, non-rigid targets. As they walk, the incidence angle of their limbs changes continuously. Compressing this dynamic 3D surface into a raw downstream scalar histogram destroys spatial topology. A simple change in walking stride radically alters the histogram bin counts, causing the tracker to artificially drop similarity scores and trigger Identity Switches (IDSW).

2. **Occlusion Contamination in Bounding Boxes:**
   In 2D camera-based DeepSORT, bounding boxes capture high-dimensional structured textures (logos, clothing patterns). In LiDAR, bounding boxes encapsulate sparse, unstructured points. When a pedestrian walks behind another person in a dense crowd, their bounding box inadvertently captures the occluder's points. A downstream tracker matching this corrupted histogram against historical memory is mathematically guaranteed to fail, often "teleporting" the ID to the wrong pedestrian.

---

## 5. Conclusion: Pushing Radiometry Upstream
The fundamental conclusion of these experiments is that **deterministic downstream algorithms (like the Hungarian solver) are ill-equipped to handle the noise and physical anisotropy of raw LiDAR material signatures.**

To successfully leverage reflectivity for tracking, the feature fusion must occur **upstream** inside the deep learning Voxel Feature Encoder (VFE). Because 3D Sparse Convolutions (like those in CenterPoint and VoxelNeXt) naturally learn spatial-radiometric relationships as topology-aware embeddings, they bypass quantization noise entirely.

Therefore, the optimal, mathematically sound evaluation strategy is to rely entirely on the upstream VFE to untangle materials and space, leaving the downstream tracker to act as a strict, physics-only kinematic gate (Pure Euclidean). This validates our pipeline and ensures that all reported HOTA and AMOTA improvements are strictly the result of superior neural architectural design.
