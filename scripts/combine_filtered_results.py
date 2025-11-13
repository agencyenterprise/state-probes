#!/usr/bin/env python3
"""
Combine filtered results from multiple batches into final train/dev files.
"""

import json
from pathlib import Path
from collections import defaultdict

# Official production batch IDs (only batches with 2000+ samples)
# We identify production batches by sample count (3000 or 2352 for train, 3414 for dev)
# Test batches have 5, 100 samples and should be excluded

def is_production_batch(metadata_file):
    """Check if a batch is a production batch based on sample count."""
    try:
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        num_samples = metadata.get('num_samples', 0)
        split = metadata.get('split', 'unknown')
        
        # Production batches:
        # - dev: 3414 samples (full dev split)
        # - train: 3000 or 2352 samples (chunked train split)
        if split == 'dev' and num_samples >= 3000:
            return True
        elif split == 'train' and num_samples >= 2000:
            return True
        
        return False
    except:
        return False


def combine_batches():
    """Combine all filtered batch results into final datasets."""
    
    batch_results_dir = Path("/workspace/state-probes/data/batch_results")
    output_dir = Path("/workspace/state-probes/data/selfie_format")
    
    # Find all filtered files
    all_filtered_files = list(batch_results_dir.glob("*_filtered_msgbatch_*.jsonl"))
    
    # Filter to only production batches (using metadata to check sample count)
    filtered_files = []
    for f in all_filtered_files:
        # Extract batch_id from filename
        batch_id = f.stem.split('_msgbatch_')[1]
        metadata_file = batch_results_dir / f'batch_metadata_msgbatch_{batch_id}.json'
        
        if is_production_batch(metadata_file):
            filtered_files.append(f)
        else:
            # Load metadata to show why it was skipped
            try:
                with open(metadata_file) as mf:
                    meta = json.load(mf)
                    print(f"Skipping test batch: {f.name} ({meta.get('num_samples')} samples)")
            except:
                print(f"Skipping: {f.name} (no metadata)")
    
    print(f"\nFound {len(filtered_files)} production filtered batch files (out of {len(all_filtered_files)} total)")
    
    # Group by split
    splits = defaultdict(list)
    
    for filtered_file in sorted(filtered_files):
        # Read the file to get samples
        samples = []
        with open(filtered_file) as f:
            for line in f:
                samples.append(json.loads(line))
        
        # Determine split from metadata
        batch_id = filtered_file.stem.split('_msgbatch_')[1]
        metadata_file = batch_results_dir / f'batch_metadata_msgbatch_{batch_id}.json'
        
        with open(metadata_file) as f:
            metadata = json.load(f)
            split = metadata.get('split', 'unknown')
        
        splits[split].extend(samples)
        print(f"  {filtered_file.name}: {len(samples)} samples ({split}, batch {batch_id[:10]}...)")
    
    # Combine and save
    print(f"\n{'='*60}")
    print(f"Combining results")
    print(f"{'='*60}")
    
    total_combined = 0
    
    for split, samples in splits.items():
        if split == 'unknown':
            continue
        
        output_file = output_dir / f"{split}_filtered.jsonl"
        
        print(f"\n{split.upper()} split:")
        print(f"  Total samples: {len(samples)}")
        print(f"  Output: {output_file}")
        
        # Write combined file
        with open(output_file, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
        
        total_combined += len(samples)
        
        # Compute statistics
        property_counts = defaultdict(int)
        entity_counts = defaultdict(int)
        
        for sample in samples:
            property_counts[sample['positive_state']] += 1
            entity_counts[sample['entity']] += 1
        
        print(f"\n  Property distribution:")
        for prop, count in sorted(property_counts.items(), key=lambda x: -x[1]):
            pct = count / len(samples) * 100
            print(f"    {prop}: {count} ({pct:.1f}%)")
        
        print(f"\n  Top 10 entities:")
        for entity, count in sorted(entity_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"    {entity}: {count}")
    
    print(f"\n{'='*60}")
    print(f"COMBINATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total filtered samples: {total_combined}")
    print(f"\nOutput files:")
    for split in splits.keys():
        if split != 'unknown':
            print(f"  - {output_dir / f'{split}_filtered.jsonl'}")

if __name__ == '__main__':
    combine_batches()

