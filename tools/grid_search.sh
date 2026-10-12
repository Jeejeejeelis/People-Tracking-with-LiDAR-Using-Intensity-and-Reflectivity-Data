#!/bin/bash

# Define parameter grid
DIST_THRESHOLDS=(20.0 30.0 40.0)
FAR_SCORES=(0.5 0.6 0.7)
REFL_WEIGHTS=(0.0 0.5 1.0 1.5 2.0)

echo "Starting Ablation Grid Search..."
echo "----------------------------------------------------"

# 1. Baseline Run (No adaptive threshold, No refl penalty)
echo "[BASELINE] No Adaptive Threshold | No Reflectivity Penalty"
python tools/tune_tracking_frontiers.py \
  --dist_thresh 999.0 --far_score 0.4 --near_score 0.4 --refl_weight 0.0 | grep "AMOTA:"

echo "----------------------------------------------------"
echo "[ABLATION 1] Frontier 3 Only (Adaptive Thresholding, refl_weight=0.0)"
for dist in "${DIST_THRESHOLDS[@]}"; do
  for far in "${FAR_SCORES[@]}"; do
    echo "Testing Dist > ${dist}m -> Score ${far}..."
    python tools/tune_tracking_frontiers.py \
      --dist_thresh ${dist} --far_score ${far} --near_score 0.4 --refl_weight 0.0 | grep "AMOTA:"
  done
done

echo "----------------------------------------------------"
echo "[ABLATION 2] Frontier 1 Only (Reflectivity Penalty, No Adaptive Threshold)"
for weight in "${REFL_WEIGHTS[@]}"; do
  if [ "$weight" != "0.0" ]; then
    echo "Testing Reflectivity Penalty Weight: ${weight}x..."
    python tools/tune_tracking_frontiers.py \
      --dist_thresh 999.0 --far_score 0.4 --near_score 0.4 --refl_weight ${weight} | grep "AMOTA:"
  fi
done

echo "----------------------------------------------------"
echo "[MASTER RUN] Combined Frontiers (Best Guesses: 30m / 0.6 / 1.5x)"
python tools/tune_tracking_frontiers.py \
  --dist_thresh 30.0 --far_score 0.6 --near_score 0.4 --refl_weight 1.5 | grep "AMOTA:"

echo "----------------------------------------------------"
echo "Grid Search Complete!"
