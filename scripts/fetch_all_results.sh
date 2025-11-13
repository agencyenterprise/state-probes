#!/bin/bash
# Fetch results from all production batches and combine them

source /tmp/state-probes-venv/bin/activate
cd /workspace/state-probes

echo "============================================"
echo "Fetching all batch results"
echo "============================================"
echo ""

# DEV SPLIT
echo "Fetching dev split..."
python scripts/validate_selfie_dataset.py --mode fetch --batch-id msgbatch_01WDxnYpQdr6qxGG4b6ydcU4

# TRAIN SPLIT BATCHES
echo ""
echo "Fetching train split batches..."

for batch_id in \
    msgbatch_01BXG27UJT5VGeF8rGWauQJF \
    msgbatch_01JyiBhTZDT5aTkWrdoSCNhi \
    msgbatch_018KdQFXMB4egk94xDdwbQZZ \
    msgbatch_018NYokgUu4KLCwbnxJr2VZq \
    msgbatch_01GbkSy2s65psEc2WxRnt11P \
    msgbatch_01Nw8KyHTx7TZXh1mDzd5u1p \
    msgbatch_01JTckcFrU2oL3LKKnh7ixsJ \
    msgbatch_01LT4HimKnEUiZ8bphFNpu43 \
    msgbatch_01Ku2mYfadcQ3jF7MR8WePhb \
    msgbatch_01AZ1CE3cyfx9Ae2Sz23Si9E \
    msgbatch_0112rsi13oWVQkGAXrB3P5Go
do
    echo ""
    echo "Fetching batch: $batch_id"
    python scripts/validate_selfie_dataset.py --mode fetch --batch-id $batch_id
done

echo ""
echo "============================================"
echo "Combining train split results..."
echo "============================================"

# Combine all train_filtered.jsonl files
cat /workspace/state-probes/data/selfie_format/train_filtered.jsonl > /workspace/state-probes/data/selfie_format/train_filtered_combined.jsonl 2>/dev/null || true

echo "All results fetched!"
echo ""
echo "Output files:"
echo "  - data/selfie_format/dev_filtered.jsonl"
echo "  - data/selfie_format/train_filtered_combined.jsonl"

