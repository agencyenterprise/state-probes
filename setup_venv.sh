#!/bin/bash
# Setup script for state-probes with PyTorch 2.x using existing virtualenv

set -e

VENV_PATH="/tmp/state-probes-venv"

echo "=========================================="
echo "State-Probes PyTorch 2.x Setup"
echo "=========================================="
echo ""

echo "Using virtual environment at $VENV_PATH..."
source $VENV_PATH/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo ""
echo "Installing PyTorch 2.x (CPU version for compatibility)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "Installing other requirements (transformers, textworld, etc.)..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source $VENV_PATH/bin/activate"
echo ""
echo "Before running any commands, set environment variables:"
echo "  export PYTHONPATH=/workspace/state-probes"
echo "  export TOKENIZERS_PARALLELISM=true"
echo ""
echo "To verify the installation:"
echo "  python test_setup.py"
echo ""
