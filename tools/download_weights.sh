#!/bin/bash
# Script to download pre-trained model weights from GitHub Releases v1.0.0
set -e

RELEASE_TAG="v1.0.0"
BASE_URL="https://github.com/Jeejeejeelis/People-Tracking-with-LiDAR-Using-Intensity-and-Reflectivity-Data/releases/download/${RELEASE_TAG}"
WEIGHTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/weights"

mkdir -p "$WEIGHTS_DIR"

CHAMPION_MODELS=(
    "hmfn_baseline_160.pth"
    "hmfn_enhanced_tuned_160.pth"
    "pfsd_hmfn_kitchen_sink_160.pth"
    "cp_baseline_tuned_160.pth"
    "cp_decoupled_160.pth"
    "pfsd_centerpoint_kitchen_sink.pth"
    "voxelnext_baseline_160.pth"
    "vn_decoupled_160.pth"
    "pfsd_voxelnext_kitchen_sink_160.pth"
    "pp_baseline_160.pth"
)

ALL_MODELS=(
    "hmfn_baseline_160.pth"
    "hmfn_decoupled_160.pth"
    "hmfn_swap_epoch_160.pth"
    "hmfn_enhanced_tuned_160.pth"
    "pfsd_hmfn_kitchen_sink_160.pth"
    "cp_baseline_tuned_160.pth"
    "cp_decoupled_160.pth"
    "pfsd_centerpoint_swap_160.pth"
    "cp_enhanced_tuned_160.pth"
    "pfsd_centerpoint_kitchen_sink.pth"
    "voxelnext_baseline_160.pth"
    "vn_decoupled_160.pth"
    "pfsd_voxelnext_swap_160.pth"
    "vn_enhanced_tuned_160.pth"
    "pfsd_voxelnext_kitchen_sink_160.pth"
    "pp_baseline_160.pth"
    "pp_decoupled_160.pth"
    "pp_swap_160.pth"
    "pp_enhanced_160.pth"
    "hmfn_pfsd.pth"
)

download_file() {
    local filename="$1"
    local dest="$WEIGHTS_DIR/$filename"
    if [ -f "$dest" ]; then
        echo "  [✓] $filename already exists, skipping."
    else
        echo "  [↓] Downloading $filename..."
        curl -L --fail --progress-bar "$BASE_URL/$filename" -o "$dest"
    fi
}

MODE="${1:-best}"

echo "=========================================================="
echo " Downloading Pre-trained Model Weights (${RELEASE_TAG})"
echo " Target directory: $WEIGHTS_DIR"
echo "=========================================================="

if [ "$MODE" == "all" ]; then
    echo "Downloading all 20 ablation checkpoints..."
    for model in "${ALL_MODELS[@]}"; do
        download_file "$model"
    done
elif [ "$MODE" == "best" ]; then
    echo "Downloading top champion models for each architecture..."
    for model in "${CHAMPION_MODELS[@]}"; do
        download_file "$model"
    done
else
    echo "Downloading single model: $MODE"
    download_file "$MODE"
fi

echo "=========================================================="
echo " Download complete! Checkpoints are in $WEIGHTS_DIR"
echo "=========================================================="
