#!/bin/bash
# Environment Setup Script for "People Tracking with LiDAR Using Intensity and Reflectivity Data"
set -e

echo "=========================================================="
echo " Setting up People Tracking with LiDAR Environment"
echo "=========================================================="

# 1. Create and activate virtual environment if not present
if [ ! -d "venv" ]; then
    echo "[+] Creating virtual environment 'venv'..."
    python3 -m venv venv
fi

echo "[+] Activating virtual environment..."
source venv/bin/activate

# 2. Upgrade pip and install requirements
echo "[+] Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Setup OpenPCDet / PFSD
echo "[+] Setting up PFSD / OpenPCDet CUDA extensions..."
cd PFSD
python setup.py develop
cd ..

# 4. Export PYTHONPATH
export PYTHONPATH=$(pwd):$(pwd)/PFSD:$(pwd)/OpenPCDet:$PYTHONPATH

echo "=========================================================="
echo " Setup complete! To activate your environment in a session:"
echo "   source venv/bin/activate"
echo "   export PYTHONPATH=\$(pwd):\$(pwd)/PFSD:\$(pwd)/OpenPCDet:\$PYTHONPATH"
echo "=========================================================="
