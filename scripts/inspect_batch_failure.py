#!/usr/bin/env python3
"""
Quick script to inspect a specific failure from batch results.
"""

import json
import sys
from pathlib import Path

def inspect_sample(batch_id, sample_idx):
    """Inspect a specific sample from a batch."""
    
    batch_dir = Path("/workspace/state-probes/data/batch_results")
    
    # Load metadata
    metadata_file = batch_dir / f"batch_metadata_{batch_id}.json"
    if not metadata_file.exists():
        print(f"Error: Batch metadata not found for {batch_id}")
        return
    
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    # Load samples
    samples_file = Path(metadata['samples_file'])
    samples = []
    with open(samples_file) as f:
        for line in f:
            samples.append(json.loads(line))
    
    if sample_idx >= len(samples):
        print(f"Error: Sample {sample_idx} out of range (max: {len(samples)-1})")
        return
    
    sample = samples[sample_idx]
    
    # Load request metadata
    request_metadata_file = Path(metadata['metadata_requests_file'])
    request_meta = None
    with open(request_metadata_file) as f:
        for line in f:
            meta = json.loads(line)
            if meta['sample_idx'] == sample_idx:
                request_meta = meta
                break
    
    # Display sample
    print(f"\n{'='*80}")
    print(f"SAMPLE #{sample_idx}")
    print(f"{'='*80}\n")
    
    print(f"Entity: {sample['entity']}")
    print(f"Expected State: {sample['positive_state']}")
    print(f"Opposite State: {sample['negative_state']}")
    
    if request_meta:
        print(f"\nPresented Options:")
        print(f"  A) {request_meta['option_a']}")
        print(f"  B) {request_meta['option_b']}")
        print(f"  Correct Answer: {request_meta['correct_choice']}")
    
    print(f"\n{'='*80}")
    print(f"CONTEXT ({len(sample['context'])} chars, {len(sample['context'].split())} words)")
    print(f"{'='*80}\n")
    print(sample['context'])
    
    # Check for explicit mentions of the state
    context_lower = sample['context'].lower()
    entity_lower = sample['entity'].lower()
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS")
    print(f"{'='*80}\n")
    
    # Find mentions of the entity
    lines_with_entity = []
    for i, line in enumerate(sample['context'].split('\n')):
        if entity_lower in line.lower():
            lines_with_entity.append((i, line))
    
    print(f"Lines mentioning '{sample['entity']}':")
    for line_num, line in lines_with_entity:
        print(f"  Line {line_num}: {line.strip()}")
    
    # Look for state keywords
    state_keywords = ['open', 'closed', 'eaten', 'locked', 'unlocked']
    print(f"\nState-related keywords near '{sample['entity']}':")
    for line_num, line in lines_with_entity:
        for keyword in state_keywords:
            if keyword in line.lower():
                print(f"  Line {line_num}: '{keyword}' found")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python inspect_batch_failure.py <batch_id> <sample_idx>")
        print("\nExample:")
        print("  python inspect_batch_failure.py msgbatch_01BYbif6L42nrY52SbR4QD2g 4")
        sys.exit(1)
    
    batch_id = sys.argv[1]
    sample_idx = int(sys.argv[2])
    
    inspect_sample(batch_id, sample_idx)

