"""
Convert TextWorld traces to SelfIE evaluation format.

Output: (context, entity, positive_state, negative_state) tuples where:
- context: Progressive narrative showing actions/observations
- entity: Specific object being tracked
- positive_state: Current ground truth state
- negative_state: Mutually exclusive contrastive state
"""

import json
import os
import glob
import random
from tqdm import tqdm
import regex as re
from collections import defaultdict

# Properties to track and their opposites
# Note: We only track properties that appear explicitly in TextWorld facts.
# Some properties (like 'unlocked', 'not eaten') are implicit (absence of the positive).
PROPERTY_OPPOSITES = {
    'open': 'closed',       # Explicit opposite in data (mutually exclusive)
    'closed': 'open',       # Explicit opposite in data (mutually exclusive)
    'locked': 'unlocked',   # 'locked' is explicit; 'unlocked' is implicit (absence)
    'eaten': 'not eaten',   # 'eaten' is explicit; 'not eaten' is implicit (absence)
}

# We only track properties that appear explicitly in the data
# (the keys, not the implicit opposites that are values)
TRACKED_PROPERTIES = set(PROPERTY_OPPOSITES.keys())


def find_mention_in_context(entity, context):
    """
    Check if entity is mentioned in the context.
    Returns True if found, False otherwise.
    """
    if entity == "player":
        entity = "you"
    
    # Use word boundaries to find entity mentions
    # Try both lowercase and title case
    entity_lower = entity.lower()
    entity_title = entity.title()
    
    # Check for word boundary before and after
    # \b doesn't work well with regex module for multi-word, so use lookahead/lookbehind
    pattern = f'(?<![a-zA-Z]){re.escape(entity_lower)}(?![a-zA-Z])'
    if re.search(pattern, context.lower()):
        return True
    
    # Also check for title-cased room headers like "-= Bedroom =-"
    if f'-= {entity_title} =-' in context:
        return True
    
    return False


def extract_entity_properties(facts):
    """
    Extract unary properties for each entity from structured facts.
    
    Args:
        facts: List of fact dictionaries from *_states.txt
        
    Returns:
        dict: entity_name -> set of properties
    """
    entity_props = defaultdict(set)
    
    for fact in facts:
        prop_name = fact.get('name', '')
        
        # Only track unary predicates (properties, not relations)
        if prop_name not in TRACKED_PROPERTIES:
            continue
            
        args = fact.get('arguments', [])
        if len(args) != 1:
            # Skip non-unary predicates
            continue
            
        entity_name = args[0].get('name', '')
        if entity_name:
            entity_props[entity_name].add(prop_name)
    
    return entity_props


def process_trace(trace_txt_path, trace_states_path):
    """
    Process a single trace file and generate SelfIE samples.
    
    Creates progressive context samples where each sample includes:
    - Growing narrative context (initial room + sequence of actions)
    - Entity states tracked after each action
    
    The states file has one state per timestep:
    - states[0] = initial state before any actions
    - states[i] = state after executing the ith action from the trace
    
    Returns:
        List of dict with keys: context, entity, positive_state, negative_state
    """
    samples = []
    
    # Load the natural language trace
    lines = []
    with open(trace_txt_path) as f:
        for line in f:
            # Stop at "The End"
            if line.strip().startswith("***") and line.strip().endswith("***"):
                break
            lines.append(line.rstrip())
    
    # Remove the task instruction (first line before room description)
    # Find the first room description line (starts with "-=")
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("-=") and line.strip().endswith("=-"):
            start_idx = i
            break
    
    # Keep only gameplay content
    lines = lines[start_idx:]
    
    # Split into action chunks (each chunk = one "> command" + its response text)
    # all_actions[0] will be the initial room description (before first ">")
    # all_actions[1..n] will be actual action commands with their responses
    all_actions = []
    curr_action = []
    
    for line in lines:
        if line.startswith(">"):
            # New action - save previous accumulated text
            if curr_action:
                all_actions.append('\n'.join(curr_action))
            curr_action = [line]
        else:
            curr_action.append(line)
    
    # Add final action if exists
    if curr_action:
        all_actions.append('\n'.join(curr_action))
    
    # Load the structured states
    states = []
    with open(trace_states_path) as f:
        for line in f:
            state = json.loads(line)
            states.append(state)
    
    if len(states) == 0 or len(all_actions) < 2:
        return samples
    
    # Create progressive contexts and samples
    # Start from context_end_idx=2 to ensure we have meaningful context
    # 
    # INDEXING EXPLANATION:
    # - all_actions[0] = initial room description (not an action)
    # - all_actions[1] = first "> command" which produces states[1]
    # - all_actions[n] = nth "> command" which produces states[n]
    # 
    # When context includes all_actions[0:k], the last action is all_actions[k-1]
    # which corresponds to states[k-1] (the state AFTER that action executed)
    #
    for context_end_idx in range(2, min(len(all_actions), len(states))):
        # Build context: includes all_actions[0] through all_actions[context_end_idx-1]
        context = '\n'.join(all_actions[:context_end_idx])
        
        if not context.strip():
            continue
        
        # Get the state AFTER the last action in context
        # Last action is at index (context_end_idx - 1), so use states[context_end_idx - 1]
        corresponding_state_idx = context_end_idx - 1
        if corresponding_state_idx >= len(states):
            continue
            
        current_state = states[corresponding_state_idx]
        
        # Extract entity properties from full_facts
        if 'full_facts' not in current_state:
            continue
        
        entity_properties = extract_entity_properties(current_state['full_facts'])
        
        # Create samples for each entity with tracked properties
        for entity_name, properties in entity_properties.items():
            # Only include if entity is mentioned in context
            if not find_mention_in_context(entity_name, context):
                continue
            
            # Create sample for each property this entity has
            for prop in properties:
                if prop not in PROPERTY_OPPOSITES:
                    continue
                
                positive_state = prop
                negative_state = PROPERTY_OPPOSITES[prop]
                
                # For explicit opposites (like open/closed), verify mutual exclusivity
                # Only need to check if the negative state is also explicitly tracked
                # (implicit opposites like 'unlocked' won't appear in properties)
                if negative_state in TRACKED_PROPERTIES and negative_state in properties:
                    # Both states present simultaneously - data quality issue, skip
                    print(f"Warning: Entity '{entity_name}' has both '{positive_state}' and '{negative_state}' - skipping")
                    continue
                
                samples.append({
                    'context': context.strip(),
                    'entity': entity_name,
                    'positive_state': positive_state,
                    'negative_state': negative_state,
                })
    
    return samples


def filter_by_length(samples, min_words=50):
    """
    Filter out samples with very short contexts.
    
    Short contexts (e.g., just initial room description) are too trivial
    because they don't test state tracking through actions.
    
    Args:
        samples: List of sample dicts
        min_words: Minimum word count for context
        
    Returns:
        Filtered list of samples
    """
    filtered = []
    for sample in samples:
        word_count = len(sample['context'].split())
        if word_count >= min_words:
            filtered.append(sample)
    
    return filtered


def balance_entity_states(samples):
    """
    Balance positive/negative states for each entity to prevent spurious correlations.
    
    For each entity, ensure roughly 50/50 distribution across its possible states.
    This prevents models from learning shortcuts like "wooden doors are usually closed".
    
    Entities that only appear with one state are removed entirely (can't be balanced).
    
    Args:
        samples: List of sample dicts
        
    Returns:
        Tuple of (balanced_samples, skipped_entities)
    """
    # Group by entity and positive_state
    entity_state_groups = defaultdict(lambda: defaultdict(list))
    
    for sample in samples:
        entity = sample['entity']
        pos_state = sample['positive_state']
        entity_state_groups[entity][pos_state].append(sample)
    
    balanced = []
    skipped_entities = []
    
    for entity, state_dict in entity_state_groups.items():
        # If entity only ever appears with one state, skip it entirely
        # (no way to balance, and model could memorize entity->state mapping)
        if len(state_dict) == 1:
            skipped_entities.append((entity, list(state_dict.keys())[0]))
            continue
        
        # Find minimum count across all states for this entity
        min_count = min(len(samples) for samples in state_dict.values())
        
        # Downsample each state to min_count to achieve balance
        for state, sample_list in state_dict.items():
            balanced.extend(random.sample(sample_list, min_count))
    
    return balanced, skipped_entities


def compute_state_change_metrics(samples, all_states_by_file):
    """
    Compute metrics about state changes vs static states.
    
    For each sample, check if the entity's state changed between initial
    state and the state corresponding to the context.
    
    Args:
        samples: List of sample dicts
        all_states_by_file: Dict mapping trace file paths to their state sequences
        
    Returns:
        Dict with metrics
    """
    # This would require tracking which state file each sample came from
    # and comparing states across timesteps. For now, return placeholder.
    return {
        'state_changed': 0,
        'state_static': 0,
        'note': 'State change tracking requires additional metadata'
    }


def process_dataset(data_dir, split, output_path, min_words=50):
    """
    Process all traces in a dataset split and apply quality filters.
    
    Args:
        data_dir: Root data directory (e.g., tw_data/simple_traces)
        split: 'train' or 'dev'
        output_path: Where to write output JSONL
        min_words: Minimum context length in words
    """
    print(f"\n{'='*60}")
    print(f"Processing {split} split")
    print(f"{'='*60}\n")
    
    split_dir = os.path.join(data_dir, split)
    state_files = sorted(glob.glob(os.path.join(split_dir, "*_states.txt")))
    
    print(f"Found {len(state_files)} trace files")
    
    # Generate all samples first
    all_samples = []
    
    for state_file in tqdm(state_files, desc=f"Generating samples from {split}"):
        # Get corresponding text file
        txt_file = state_file.replace('_states.txt', '.txt')
        
        if not os.path.exists(txt_file):
            print(f"Warning: Missing text file for {state_file}")
            continue
        
        # Process this trace
        samples = process_trace(txt_file, state_file)
        all_samples.extend(samples)
    
    print(f"\n{'='*60}")
    print(f"Filtering samples for {split}")
    print(f"{'='*60}")
    print(f"Initial samples generated: {len(all_samples)}")
    
    # Apply length filter
    length_filtered = filter_by_length(all_samples, min_words=min_words)
    print(f"After length filter (>={min_words} words): {len(length_filtered)} ({len(length_filtered)/len(all_samples)*100:.1f}%)")
    
    # Apply entity-state balance filter
    balanced_samples, skipped_entities = balance_entity_states(length_filtered)
    print(f"After entity-state balancing: {len(balanced_samples)} ({len(balanced_samples)/len(length_filtered)*100:.1f}%)")
    print(f"Entities removed (only one state): {len(skipped_entities)}")
    
    # Show some examples of skipped entities
    if skipped_entities:
        print(f"\nExample skipped entities (first 5):")
        for entity, state in skipped_entities[:5]:
            print(f"  {entity}: only '{state}'")
    
    # Use balanced samples for output and statistics
    final_samples = balanced_samples
    
    # Write output
    print(f"\n{'='*60}")
    print(f"Writing {len(final_samples)} samples to {output_path}")
    print(f"{'='*60}")
    with open(output_path, 'w') as f:
        for sample in final_samples:
            f.write(json.dumps(sample) + '\n')
    
    # Compute statistics on final samples
    property_counts = defaultdict(int)
    entity_counts = defaultdict(int)
    entity_state_counts = defaultdict(lambda: defaultdict(int))
    
    for sample in final_samples:
        property_counts[sample['positive_state']] += 1
        entity_counts[sample['entity']] += 1
        entity_state_counts[sample['entity']][sample['positive_state']] += 1
    
    # Print statistics
    print(f"\nStatistics for {split}")
    print(f"{'='*60}")
    print(f"Total samples: {len(final_samples)}")
    
    print(f"\nSamples per property:")
    for prop, count in sorted(property_counts.items(), key=lambda x: -x[1]):
        print(f"  {prop}: {count}")
    
    print(f"\nTop 10 entities by sample count:")
    for entity, count in sorted(entity_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {entity}: {count}")
    
    # Context length statistics
    if final_samples:
        context_lengths = [len(s['context'].split()) for s in final_samples]
        avg_len = sum(context_lengths) / len(context_lengths)
        print(f"\nContext length statistics:")
        print(f"  Average: {avg_len:.1f} words")
        print(f"  Min: {min(context_lengths)} words")
        print(f"  Max: {max(context_lengths)} words")
    
    # Property balance
    print(f"\nProperty distribution:")
    total = sum(property_counts.values())
    for prop, count in sorted(property_counts.items()):
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {prop}: {pct:.1f}%")
    
    # Entity-state balance check (verify our balancing worked)
    print(f"\nEntity-state balance (top 10 entities):")
    for entity, count in sorted(entity_counts.items(), key=lambda x: -x[1])[:10]:
        states = entity_state_counts[entity]
        state_str = ", ".join([f"{state}={cnt}" for state, cnt in sorted(states.items())])
        print(f"  {entity}: {state_str}")
    
    return len(final_samples)


def main():
    # Configuration
    data_dir = "/workspace/state-probes/tw_data/simple_traces"
    output_dir = "/workspace/state-probes/data/selfie_format"
    min_words = 50  # Minimum context length to ensure non-trivial examples
    random_seed = 42  # For reproducibility of balancing
    
    # Set random seed for reproducible sampling
    random.seed(random_seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Configuration:")
    print(f"  Data directory: {data_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Min context length: {min_words} words")
    print(f"  Random seed: {random_seed}")
    
    # Process both splits
    train_count = process_dataset(
        data_dir=data_dir,
        split='train',
        output_path=os.path.join(output_dir, 'train.jsonl'),
        min_words=min_words
    )
    
    dev_count = process_dataset(
        data_dir=data_dir,
        split='dev',
        output_path=os.path.join(output_dir, 'dev.jsonl'),
        min_words=min_words
    )
    
    # Overall summary
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Train samples: {train_count}")
    print(f"Dev samples: {dev_count}")
    print(f"Total samples: {train_count + dev_count}")
    print(f"\nTarget: 12,900 total (9,500 train, 3,400 dev)")
    
    if train_count + dev_count < 12900:
        print(f"\n⚠️  Warning: Generated {train_count + dev_count} samples, below target of 12,900")
        print(f"   Consider: expanding property types or adjusting filtering criteria")
    else:
        print(f"\n✓ Successfully generated sufficient samples!")
    
    print(f"\nOutput files:")
    print(f"  - {output_dir}/train.jsonl")
    print(f"  - {output_dir}/dev.jsonl")


if __name__ == '__main__':
    main()

