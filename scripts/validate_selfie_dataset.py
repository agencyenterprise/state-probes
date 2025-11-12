"""
Validate SelfIE dataset by testing samples with Claude Haiku.

This script:
1. Loads random samples from the generated dataset
2. Presents context + entity to Claude Haiku
3. Compares Haiku's response to ground truth
4. Reports accuracy and shows failure cases
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import anthropic
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Property mappings for validation
PROPERTY_OPPOSITES = {
    'open': 'closed',
    'closed': 'open',
    'locked': 'unlocked',
    'eaten': 'not eaten',
}

# All valid states (both explicit and implicit)
ALL_VALID_STATES = set(PROPERTY_OPPOSITES.keys()) | set(PROPERTY_OPPOSITES.values())


def load_samples(jsonl_path: str, num_samples: int = None, property_filter: str = None) -> List[Dict]:
    """
    Load samples from JSONL file.
    
    Args:
        jsonl_path: Path to the JSONL file
        num_samples: Number of samples to return (None for all)
        property_filter: Filter samples by property (e.g., 'open', 'locked', 'eaten')
    """
    samples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            
            # Filter by property if specified
            if property_filter:
                # Check if either positive or negative state matches the property or its opposite
                pos_state = sample['positive_state']
                neg_state = sample['negative_state']
                
                # Normalize property filter to handle both states in a pair
                if property_filter in PROPERTY_OPPOSITES:
                    filter_pair = {property_filter, PROPERTY_OPPOSITES[property_filter]}
                else:
                    # If the filter isn't in the dict, it might be a value (like 'unlocked')
                    # Find its key
                    filter_pair = {property_filter}
                    for key, val in PROPERTY_OPPOSITES.items():
                        if val == property_filter:
                            filter_pair.add(key)
                            break
                
                # Include sample if it uses this property pair
                if pos_state in filter_pair or neg_state in filter_pair:
                    samples.append(sample)
            else:
                samples.append(sample)
    
    if num_samples and num_samples < len(samples):
        samples = random.sample(samples, num_samples)
    
    return samples


def create_validation_prompt(context: str, entity: str, option_a: str, option_b: str) -> str:
    """
    Create a prompt for Claude Haiku to determine entity state.
    
    The prompt presents two mutually exclusive options and asks Haiku to choose.
    The order of options should be randomized by the caller.
    """
    prompt = f"""You are analyzing a text-based game scenario. Based on the context below, determine the current state of a specific entity.

Context:
{context}

Entity: {entity}

Which of these two states best describes "{entity}" based on the context above?

A) {option_a}
B) {option_b}

Respond with ONLY the letter (A or B) followed by the state. For example: "A" or "B: open"

Answer:"""
    
    return prompt


def extract_choice_from_response(response: str) -> str:
    """
    Extract the choice (A or B) from Haiku's response.
    
    Returns 'A' or 'B', or None if cannot be parsed.
    """
    # Clean the response
    response = response.strip().upper()
    
    # Look for A or B at the start
    if response.startswith('A'):
        return 'A'
    elif response.startswith('B'):
        return 'B'
    
    # Look for A or B anywhere in the response
    if 'A' in response and 'B' not in response:
        return 'A'
    elif 'B' in response and 'A' not in response:
        return 'B'
    
    return None


def validate_sample(client: anthropic.Anthropic, sample: Dict, model: str = "claude-haiku-4-5") -> Tuple[bool, str, str]:
    """
    Validate a single sample by asking Haiku to choose between two options.
    
    Returns:
        (is_correct, predicted_state, expected_state)
    """
    context = sample['context']
    entity = sample['entity']
    positive_state = sample['positive_state']
    negative_state = sample['negative_state']
    
    # Randomize order of options to avoid bias
    if random.random() < 0.5:
        option_a = positive_state
        option_b = negative_state
        correct_choice = 'A'
    else:
        option_a = negative_state
        option_b = positive_state
        correct_choice = 'B'
    
    prompt = create_validation_prompt(context, entity, option_a, option_b)
    
    try:
        message = client.messages.create(
            model=model,
            max_tokens=50,
            temperature=0,  # Deterministic
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        response_text = message.content[0].text
        predicted_choice = extract_choice_from_response(response_text)
        
        if predicted_choice is None:
            return False, f"UNPARSEABLE: {response_text}", positive_state
        
        # Determine which state was chosen
        predicted_state = option_a if predicted_choice == 'A' else option_b
        
        # Check if prediction matches positive state
        is_correct = (predicted_choice == correct_choice)
        
        return is_correct, predicted_state, positive_state
        
    except Exception as e:
        print(f"Error calling API: {e}")
        return False, f"ERROR: {e}", positive_state


def main():
    """Main validation routine."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate SelfIE dataset with Claude Haiku")
    parser.add_argument(
        '--split',
        type=str,
        default='dev',
        choices=['train', 'dev'],
        help='Which split to validate'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of samples to test (default: 100)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='claude-haiku-4-5',
        help='Model to use for validation'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sample selection'
    )
    parser.add_argument(
        '--property',
        type=str,
        default=None,
        choices=['open', 'closed', 'locked', 'unlocked', 'eaten', 'not eaten'],
        help='Filter samples by property type (e.g., "open" for open/closed pairs)'
    )
    parser.add_argument(
        '--show-all-failures',
        action='store_true',
        help='Show all failure cases (default: first 10)'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Setup paths
    data_dir = Path("/workspace/state-probes/data/selfie_format")
    jsonl_path = data_dir / f"{args.split}.jsonl"
    
    if not jsonl_path.exists():
        print(f"Error: Dataset file not found: {jsonl_path}")
        return
    
    # Initialize Anthropic client
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found in environment")
        return
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # Load samples
    print(f"\n{'='*60}")
    print(f"Loading samples from {args.split} split")
    if args.property:
        print(f"Filtering by property: {args.property}")
    print(f"{'='*60}")
    
    samples = load_samples(str(jsonl_path), args.num_samples, args.property)
    
    if args.property:
        print(f"Loaded {len(samples)} samples with property '{args.property}'")
    else:
        print(f"Loaded {len(samples)} samples")
    
    # Validate samples
    print(f"\n{'='*60}")
    print(f"Validating with {args.model}")
    print(f"{'='*60}\n")
    
    results = []
    correct_count = 0
    failure_cases = []
    
    for i, sample in enumerate(tqdm(samples, desc="Validating")):
        is_correct, predicted, expected = validate_sample(client, sample, model=args.model)
        
        results.append({
            'sample': sample,
            'is_correct': is_correct,
            'predicted': predicted,
            'expected': expected
        })
        
        if is_correct:
            correct_count += 1
        else:
            failure_cases.append({
                'sample_idx': i,
                'sample': sample,
                'predicted': predicted,
                'expected': expected
            })
    
    # Compute statistics
    accuracy = (correct_count / len(samples)) * 100 if samples else 0
    
    # Print results
    print(f"\n{'='*60}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Total samples tested: {len(samples)}")
    print(f"Correct: {correct_count}")
    print(f"Incorrect: {len(samples) - correct_count}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    # Property-wise breakdown
    property_stats = {}
    for result in results:
        prop = result['expected']
        if prop not in property_stats:
            property_stats[prop] = {'correct': 0, 'total': 0}
        property_stats[prop]['total'] += 1
        if result['is_correct']:
            property_stats[prop]['correct'] += 1
    
    print(f"\n{'='*60}")
    print(f"Accuracy by Property")
    print(f"{'='*60}")
    for prop in sorted(property_stats.keys()):
        stats = property_stats[prop]
        prop_acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  {prop:15s}: {stats['correct']:3d}/{stats['total']:3d} ({prop_acc:.1f}%)")
    
    # Show failure cases
    if failure_cases:
        max_failures = len(failure_cases) if args.show_all_failures else min(10, len(failure_cases))
        
        print(f"\n{'='*60}")
        print(f"Failure Cases (showing {max_failures} of {len(failure_cases)})")
        print(f"{'='*60}\n")
        
        for i, failure in enumerate(failure_cases[:max_failures]):
            sample = failure['sample']
            print(f"--- Failure #{i+1} (Sample #{failure['sample_idx']}) ---")
            print(f"Entity: {sample['entity']}")
            print(f"Expected: {failure['expected']}")
            print(f"Predicted: {failure['predicted']}")
            print(f"\nContext (full):")
            print(sample['context'])
            print()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    
    if accuracy >= 90:
        print(f"✓ EXCELLENT: {accuracy:.1f}% accuracy - dataset is high quality!")
    elif accuracy >= 70:
        print(f"⚠️  GOOD: {accuracy:.1f}% accuracy - dataset is mostly good, some issues")
    else:
        print(f"✗ POOR: {accuracy:.1f}% accuracy - dataset has significant issues")
    
    if len(failure_cases) > 0:
        print(f"\nRecommendations:")
        print(f"  - Review failure cases above")
        print(f"  - Check if context provides enough information")
        print(f"  - Verify entity mentions and state changes are clear")
    
    print(f"\nRun with --show-all-failures to see all {len(failure_cases)} failures")


if __name__ == '__main__':
    main()

