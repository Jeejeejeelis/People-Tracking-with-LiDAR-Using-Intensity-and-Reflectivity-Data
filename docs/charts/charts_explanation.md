# LiDAR Pedestrian Tracking & Ablation Charts Guide

This document explains the purpose and significance of each analytical chart generated for the master's thesis. These charts visualize the findings from the comprehensive ablation study evaluating different 3D object detection architectures (PointPillars, CenterPoint, VoxelNeXt, HMFN) and feature engineering variants (Baseline, Decoupled, Swap) on high-density pedestrian tracking.

## Core Thesis Figures

### 1. `1_2d_compression_failure.pdf`
**Title:** The 2D Compression Failure: False Positives vs Recall across Architectures
**What it shows:** A visual comparison of False Positives vs. Recall across different models. It specifically highlights the massive False Positive explosion observed in PointPillars compared to true 3D voxel models.
**Why it's important:** It serves as the "Negative Control" proof for the thesis. It visually demonstrates that 2D Z-axis compression architectures fail mathematically on dense 128-beam pedestrian crowds because they cannot separate vertical noise (like poles or signs) from pedestrians, forcing a brutal trade-off between recall and false positives.

### 2. `2_reflectivity_paradox.pdf`
**Title:** The Reflectivity Paradox: Physics Equation Drop-off vs Decoupled Triumph
**What it shows:** A grouped bar chart comparing the AMOTA metric for Baseline, Swap (hardcoded $I \times d^2$), and Decoupled (independent raw features) variants across the model architectures.
**Why it's important:** Proves the core physical hypothesis. The hardcoded Swap variant uniformly destroys performance across architectures, proving that raw intensity contains crucial non-Lambertian discriminative signatures that the physical equation corrupts. Conversely, letting pure 3D sparse networks learn the calibration natively (Decoupled) yields massive improvements.

### 3. `3_tracker_rigidity_tradeoff.pdf`
**Title:** The Tracker Rigidity Trade-off: Identity Continuity (IDF1) vs Total Tracking (AMOTA)
**What it shows:** A scatter plot mapping IDF1 against AMOTA for different tracker tuning configurations.
**Why it's important:** It visualizes the "Frankenstein Tracker" dilemma. It proves that simply optimizing for high AMOTA (tracking overlap volume) often mathematically destroys IDF1 (true identity continuity over time), highlighting the necessity of multi-metric evaluation.

### 4. `4_detection_tracking_divergence.pdf`
**Title:** Detection vs. Tracking Divergence: High Detection Does Not Guarantee Tracking
**What it shows:** A slope chart linking raw 3D Detection (mAP) to final Tracking performance (AMOTA).
**Why it's important:** It demonstrates the non-linear relationship between detection and tracking. A model with an 83% AP might yield a worse tracker than a 75% AP model due to the specific nature of its False Positive noise, which the Kalman Filter tracker incorrectly stitches into "ghost tracks".

### 5. `5_optimal_frontier_heatmaps.pdf`
**Title:** Optimal Frontier Heatmap: Stability Basins vs Reflectivity Penalty
**What it shows:** 2D heatmaps (Score Threshold vs. Reflectivity Penalty) colored by AMOTA, parsed directly from the Triton grid search logs.
**Why it's important:** Proves the existence of "stability basins" for decoupled models, showing that robust models maintain high tracking accuracy across a wide range of hyperparameter configurations, whereas fragile variants easily collapse.

---

## Metric Analysis & Advanced Trade-offs

### 6. `6_metric_sensitivity_analysis.pdf`
**What it shows:** How sensitive the final metrics (AMOTA, MOTA, IDF1) are to small perturbations in the confidence threshold and matching distance.
**Why it's important:** Demonstrates which architectures are robust for real-world deployment without requiring pinpoint hyperparameter tuning.

### 7. `7_iso_hota_metric_plane.pdf`
**Title:** The Zero-Sum Tracking Paradox: Iso-HOTA Metric Plane
**What it shows:** A mapping of tracking metrics along an Iso-HOTA (Higher Order Tracking Accuracy) plane.
**Why it's important:** Visualizes how HOTA balances detection accuracy and association accuracy, and how certain tuning hacks trade one exclusively for the other while keeping HOTA static.

### 8. `8_dual_axis_crossover.pdf`
**Title:** Dual-Axis Crossover: Strict Physics Penalty Purges Noise at the Expense of Volume
**What it shows:** A dual-axis line chart showing how increasing reflectivity penalties drastically reduces False Positives but simultaneously tanks True Positives.
**Why it's important:** Explains the mechanics behind the Swap variant's failure, showing the mathematical crossover point where the penalty becomes too aggressive for the network to maintain track volume.

### 9. `9_hota_subcomponent_slopegraph.pdf`
**Title:** HOTA Sub-Component Divergence: The "X-Shape" Trade-off of Physical Reflectivity
**What it shows:** Parallel coordinates linking DetA (Detection Accuracy) and AssA (Association Accuracy).
**Why it's important:** Breaks down HOTA to show that physical reflectivity calibration (Swap) might occasionally improve AssA (by filtering out noise) but irreparably damages DetA (by deleting valid pedestrian signatures), creating an "X-shape" divergence graph.

---

## Macro Trends & Final Comparisons

### 10. `10_hyperparameter_stability_basins.pdf`
**What it shows:** Multi-model heatmaps of stability basins across the hyperparameter grid.
**Why it's important:** Extends Chart 5 to compare stability between the State-of-the-Art (HMFN) and baseline architectures.

### 11. `11_variant_macro_trends.pdf`
**What it shows:** Broad statistical trends across the three feature variants (Baseline, Swap, Decoupled) aggregated across all tested thresholds.
**Why it's important:** Provides high-level statistical confidence that the Decoupled approach is superior on average, ignoring local optimal anomalies.

### 12. `12_fitted_optimum_curves.pdf`
**What it shows:** Polynomial fitted curves modeling the expected maximum performance of each architecture based on the grid search data.
**Why it's important:** Helps abstract away the noise of individual grid search points to show the true underlying capability curve of the models.

### 13. `13_reflectivity_penalty_degradation.pdf`
**What it shows:** The isolated impact of the reflectivity penalty hyperparameter on AMOTA degradation.
**Why it's important:** Quantifies exactly how much harm artificial physics constraints cause to the tracking logic.

### 14. `14_final_architecture_comparison.pdf`
**Title:** Absolute Best AMOTA per Model Architecture & Feature Variant
**What it shows:** The ultimate "scoreboard" bar chart comparing the absolute highest AMOTA achieved by each model/variant combination after total grid search optimization.
**Why it's important:** This is the primary results chart for the thesis conclusion, clearly crowning HMFN Baseline and CenterPoint/VoxelNeXt Decoupled as the top performers, and highlighting PointPillars' complete failure.

### 15. `15_final_architecture_comparison_hota.pdf` & 16. `16_final_architecture_comparison_idf1.pdf`
**What they show:** The equivalent absolute best scoreboards, but specifically evaluated for HOTA (Overall Tracking Capability) and IDF1 (Identity Continuity) instead of AMOTA.
**Why it's important:** Ensures the conclusion isn't biased by AMOTA's volume-heavy calculation. It confirms whether the top architectures truly maintain consistent pedestrian identities (IDF1) and balanced detection/association accuracy (HOTA).
