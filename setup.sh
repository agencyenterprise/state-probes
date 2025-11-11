#!/bin/bash
# Setup script for state-probes with PyTorch 2.x

set -e

echo "Creating conda environment..."
conda create -n state-probes python=3.9 -y

echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate state-probes

echo "Installing PyTorch 2.x (CPU version)..."
conda install pytorch cpuonly -c pytorch -y

echo "Installing other requirements..."
pip install -r requirements.txt

echo ""
echo "Setup complete! To activate the environment, run:"
echo "  conda activate state-probes"
echo ""
echo "Before running any commands, set environment variables:"
echo "  export PYTHONPATH=."
echo "  export TOKENIZERS_PARALLELISM=true"

