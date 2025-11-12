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
from tqdm import tqdm
import regex as re
from collections import defaultdict

# Properties to track and their opposites
# Note: We only track properties that appear explicitly in TextWorld facts.
# Some properties (like 'unlocked', 'not edible') are implicit (absence of the positive).
PROPERTY_OPPOSITES = {
    'open': 'closed',       # Explicit opposite in data (mutually exclusive)
    'closed': 'open',       # Explicit opposite in data (mutually exclusive)
    'locked': 'unlocked',   # 'locked' is explicit; 'unlocked' is implicit (absence)
    'edible': 'not edible', # 'edible' is explicit; 'not edible' is implicit (absence)
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


def process_dataset(data_dir, split, output_path):
    """
    Process all traces in a dataset split.
    
    Args:
        data_dir: Root data directory (e.g., tw_data/simple_traces)
        split: 'train' or 'dev'
        output_path: Where to write output JSONL
    """
    print(f"\n{'='*60}")
    print(f"Processing {split} split")
    print(f"{'='*60}\n")
    
    split_dir = os.path.join(data_dir, split)
    state_files = sorted(glob.glob(os.path.join(split_dir, "*_states.txt")))
    
    print(f"Found {len(state_files)} trace files")
    
    all_samples = []
    property_counts = defaultdict(int)
    entity_counts = defaultdict(int)
    
    for state_file in tqdm(state_files, desc=f"Processing {split}"):
        # Get corresponding text file
        txt_file = state_file.replace('_states.txt', '.txt')
        
        if not os.path.exists(txt_file):
            print(f"Warning: Missing text file for {state_file}")
            continue
        
        # Process this trace
        samples = process_trace(txt_file, state_file)
        
        # Collect statistics
        for sample in samples:
            property_counts[sample['positive_state']] += 1
            entity_counts[sample['entity']] += 1
        
        all_samples.extend(samples)
    
    # Write output
    print(f"\nWriting {len(all_samples)} samples to {output_path}")
    with open(output_path, 'w') as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + '\n')
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"Statistics for {split}")
    print(f"{'='*60}")
    print(f"Total samples: {len(all_samples)}")
    print(f"\nSamples per property:")
    for prop, count in sorted(property_counts.items(), key=lambda x: -x[1]):
        print(f"  {prop}: {count}")
    
    print(f"\nTop 10 entities:")
    for entity, count in sorted(entity_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {entity}: {count}")
    
    # Context length statistics
    if all_samples:
        context_lengths = [len(s['context'].split()) for s in all_samples]
        avg_len = sum(context_lengths) / len(context_lengths)
        print(f"\nAverage context length: {avg_len:.1f} words")
        print(f"Min context length: {min(context_lengths)} words")
        print(f"Max context length: {max(context_lengths)} words")
    
    # Property balance
    print(f"\nProperty distribution:")
    total = sum(property_counts.values())
    for prop, count in sorted(property_counts.items()):
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {prop}: {pct:.1f}%")
    
    return len(all_samples)


def main():
    # Configuration
    data_dir = "/workspace/state-probes/tw_data/simple_traces"
    output_dir = "/workspace/state-probes/data/selfie_format"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process both splits
    train_count = process_dataset(
        data_dir=data_dir,
        split='train',
        output_path=os.path.join(output_dir, 'train.jsonl')
    )
    
    dev_count = process_dataset(
        data_dir=data_dir,
        split='dev',
        output_path=os.path.join(output_dir, 'dev.jsonl')
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

