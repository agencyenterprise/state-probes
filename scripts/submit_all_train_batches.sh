#!/bin/bash
# Submit all train split batches in chunks of 3000

source /tmp/state-probes-venv/bin/activate
cd /workspace/state-probes

# Total: 32,352 samples, using chunks of 3000
# Batch 1 (0-2999) already submitted

echo "Submitting train split batches 2-11..."

for i in {3000..30000..3000}; do
    echo ""
    echo "========================================="
    echo "Submitting batch starting at index $i"
    echo "========================================="
    python scripts/validate_selfie_dataset.py --mode submit --split train --start-idx $i --batch-size 3000
    sleep 2  # Small delay between submissions
done

# Final batch (30000-32351)
echo ""
echo "========================================="
echo "Submitting final batch (30000-32351)"
echo "========================================="
python scripts/validate_selfie_dataset.py --mode submit --split train --start-idx 30000 --batch-size 3000

echo ""
echo "========================================="
echo "All train batches submitted!"
echo "========================================="
echo "Submitted 11 train batches total"
echo "Check back in a few hours to fetch results"

