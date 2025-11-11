#!/bin/bash
# Quick activation script for state-probes environment

# Activate virtual environment
source /tmp/state-probes-venv/bin/activate

# Set required environment variables
export PYTHONPATH=/workspace/state-probes
export TOKENIZERS_PARALLELISM=true

echo "✓ Environment activated!"
echo ""
echo "Virtual env: /tmp/state-probes-venv"
echo "PYTHONPATH: $PYTHONPATH"
echo "TOKENIZERS_PARALLELISM: $TOKENIZERS_PARALLELISM"
echo ""
echo "You can now run scripts like:"
echo "  python test_setup.py"
echo "  python scripts/train_textworld.py --help"
echo "  python scripts/probe_textworld.py --help"
echo ""

