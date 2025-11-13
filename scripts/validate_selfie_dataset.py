"""
Validate and filter SelfIE dataset using Claude Haiku via Batch API.

This script:
1. Loads samples from the generated dataset
2. Submits batch requests to Claude Haiku asking:
   - Is the answer solvable from the context?
   - What is the answer?
3. Fetches results and creates filtered dataset
4. Reports accuracy and shows failure cases

Usage:
  # Submit test batch
  python validate_selfie_dataset.py --mode submit --num-samples 100 --split dev
  
  # Submit full dataset
  python validate_selfie_dataset.py --mode submit --split dev
  
  # Fetch results and create filtered dataset
  python validate_selfie_dataset.py --mode fetch --batch-id <batch-id>
"""

import json
import os
import random
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
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
    
    Asks two questions:
    1. Is the state solvable from the context?
    2. What is the answer (A or B)?
    
    Expects JSON response: {"solvable": bool, "answer": "A"|"B"|null}
    """
    prompt = f"""You are analyzing a text-based game scenario. Based on the context below, determine the current state of a specific entity.

Context:
{context}

Entity: {entity}

Which of these two states best describes "{entity}" based on the context above?

A) {option_a}
B) {option_b}

Answer these two questions:
1. Can you determine the answer from the information provided in the context?
2. If yes, which option (A or B) is correct?

You may reason through the problem first if helpful, but you must end your response with a JSON object in this exact format:
{{"solvable": true, "answer": "A"}}

OR if the answer cannot be determined:
{{"solvable": false, "answer": null}}"""
    
    return prompt


def parse_json_response(response: str) -> Tuple[Optional[bool], Optional[str]]:
    """
    Parse JSON response from model.
    
    Expected format: {"solvable": bool, "answer": "A"|"B"|null}
    
    The model may provide reasoning before the JSON. We search for the JSON
    object anywhere in the response.
    
    Returns:
        (solvable, answer) where:
        - solvable: True/False/None (None if parse error)
        - answer: "A"/"B"/None
    """
    try:
        response = response.strip()
        
        # Search for JSON object in the response
        # Look for the pattern {"solvable": ...}
        import re
        
        # First try to extract from ```json code block if present
        json_block_match = re.search(r'```json\s*(\{[^`]+\})\s*```', response, re.DOTALL)
        if json_block_match:
            response = json_block_match.group(1).strip()
        
        # Try to find JSON object with a simpler pattern
        # Look for {"solvable": ... "answer": ...} anywhere in the text
        json_match = re.search(r'\{[^{}]*"solvable"[^{}]*"answer"[^{}]*\}', response)
        
        if json_match:
            json_str = json_match.group(0)
            try:
                data = json.loads(json_str)
                if 'solvable' in data and 'answer' in data:
                    return data.get('solvable'), data.get('answer')
            except json.JSONDecodeError:
                pass
        
        # Try direct parsing (in case the whole response is JSON)
        try:
            data = json.loads(response)
            solvable = data.get('solvable')
            answer = data.get('answer')
            return solvable, answer
        except json.JSONDecodeError:
            pass
        
        # Try to find any JSON object with curly braces
        brace_start = response.find('{')
        if brace_start != -1:
            # Find matching closing brace
            brace_count = 0
            for i in range(brace_start, len(response)):
                if response[i] == '{':
                    brace_count += 1
                elif response[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = response[brace_start:i+1]
                        try:
                            data = json.loads(json_str)
                            if 'solvable' in data and 'answer' in data:
                                return data.get('solvable'), data.get('answer')
                        except json.JSONDecodeError:
                            pass
                        break
        
        return None, None
        
    except Exception as e:
        return None, None


def prepare_batch_request(sample: Dict, sample_idx: int, model: str = "claude-haiku-4-5") -> tuple[Dict, Dict]:
    """
    Prepare a single sample for batch API submission.
    
    Returns (batch_request, metadata_dict) where:
    - batch_request: dict in format required by Anthropic's Batch API
    - metadata_dict: extra info needed for evaluation (stored separately)
    """
    context = sample['context']
    entity = sample['entity']
    positive_state = sample['positive_state']
    negative_state = sample['negative_state']
    
    # Randomize order of options to avoid bias
    # Use sample_idx as seed for reproducibility
    random.seed(sample_idx)
    if random.random() < 0.5:
        option_a = positive_state
        option_b = negative_state
        correct_choice = 'A'
    else:
        option_a = negative_state
        option_b = positive_state
        correct_choice = 'B'
    
    prompt = create_validation_prompt(context, entity, option_a, option_b)
    
    batch_request = {
        "custom_id": str(sample_idx),
        "params": {
            "model": model,
            "max_tokens": 500,  # Increased to allow for reasoning before JSON
            "temperature": 0,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    }
    
    metadata = {
        "sample_idx": sample_idx,
        "entity": entity,
        "positive_state": positive_state,
        "negative_state": negative_state,
        "option_a": option_a,
        "option_b": option_b,
        "correct_choice": correct_choice
    }
    
    return batch_request, metadata


def submit_batch(client: anthropic.Anthropic, samples: List[Dict], 
                 output_dir: Path, model: str = "claude-haiku-4-5", split: str = None) -> str:
    """
    Submit samples to Batch API.
    
    Args:
        client: Anthropic API client
        samples: List of samples to process
        output_dir: Directory to store batch files
        model: Model name to use
        split: Original split name ('train' or 'dev')
    
    Returns batch_id.
    """
    # Create batch requests
    print(f"\nPreparing {len(samples)} samples for batch submission...")
    batch_requests = []
    request_metadata = []
    for idx, sample in enumerate(tqdm(samples, desc="Preparing requests")):
        request, metadata = prepare_batch_request(sample, idx, model)
        batch_requests.append(request)
        request_metadata.append(metadata)
    
    # Write requests to JSONL file
    requests_file = output_dir / f"batch_requests_{int(time.time())}.jsonl"
    print(f"\nWriting batch requests to {requests_file}...")
    with open(requests_file, 'w') as f:
        for req in batch_requests:
            f.write(json.dumps(req) + '\n')
    
    # Save metadata separately
    metadata_requests_file = output_dir / f"batch_request_metadata_{requests_file.stem}.jsonl"
    print(f"Saving request metadata to {metadata_requests_file}...")
    with open(metadata_requests_file, 'w') as f:
        for meta in request_metadata:
            f.write(json.dumps(meta) + '\n')
    
    # Also save samples for later reference
    samples_file = output_dir / f"batch_samples_{requests_file.stem}.jsonl"
    print(f"Saving samples to {samples_file}...")
    with open(samples_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    # Submit to Batch API
    print(f"\nSubmitting batch to Anthropic API...")
    
    # Read and parse the requests
    with open(requests_file, 'r') as f:
        requests_data = [json.loads(line) for line in f]
    
    batch = client.messages.batches.create(
        requests=requests_data
    )
    
    batch_id = batch.id
    
    # Save batch metadata
    metadata_file = output_dir / f"batch_metadata_{batch_id}.json"
    batch_metadata = {
        "batch_id": batch_id,
        "requests_file": str(requests_file),
        "metadata_requests_file": str(metadata_requests_file),
        "samples_file": str(samples_file),
        "num_samples": len(samples),
        "model": model,
        "split": split,
        "submitted_at": time.time(),
        "status": batch.processing_status
    }
    with open(metadata_file, 'w') as f:
        json.dump(batch_metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Batch submitted successfully!")
    print(f"{'='*60}")
    print(f"Batch ID: {batch_id}")
    print(f"Status: {batch.processing_status}")
    print(f"Requests: {batch.request_counts}")
    print(f"\nMetadata saved to: {metadata_file}")
    print(f"\nTo fetch results, run:")
    print(f"  python validate_selfie_dataset.py --mode fetch --batch-id {batch_id}")
    
    return batch_id


def fetch_batch_results(client: anthropic.Anthropic, batch_id: str, output_dir: Path) -> Dict:
    """
    Fetch batch results and analyze them.
    
    Returns dict with results and statistics.
    """
    print(f"\n{'='*60}")
    print(f"Fetching batch results for: {batch_id}")
    print(f"{'='*60}\n")
    
    # Load batch metadata
    batch_metadata_file = output_dir / f"batch_metadata_{batch_id}.json"
    batch_metadata = None
    samples_file = None
    request_metadata_file = None
    
    if batch_metadata_file.exists():
        with open(batch_metadata_file, 'r') as f:
            batch_metadata = json.load(f)
            samples_file = Path(batch_metadata['samples_file'])
            request_metadata_file = Path(batch_metadata.get('metadata_requests_file', ''))
    else:
        print(f"Warning: Metadata file not found, will search for samples file...")
        # Try to find samples file
        for f in output_dir.glob("batch_samples_*.jsonl"):
            if batch_id in str(f):
                samples_file = f
                break
    
    # Get batch status
    batch = client.messages.batches.retrieve(batch_id)
    
    print(f"Status: {batch.processing_status}")
    print(f"Request counts: {batch.request_counts}")
    
    if batch.processing_status != "ended":
        print(f"\n⚠️  Batch is not complete yet (status: {batch.processing_status})")
        print(f"Please wait and try again later.")
        return None
    
    # Load original samples
    if not samples_file or not samples_file.exists():
        print(f"Error: Cannot find samples file for batch {batch_id}")
        print(f"Looked for: {samples_file}")
        return None
    
    print(f"\nLoading original samples from {samples_file}...")
    original_samples = []
    with open(samples_file, 'r') as f:
        for line in f:
            original_samples.append(json.loads(line))
    
    print(f"Loaded {len(original_samples)} original samples")
    
    # Load request metadata
    request_metadata_list = {}
    if request_metadata_file and request_metadata_file.exists():
        print(f"Loading request metadata from {request_metadata_file}...")
        with open(request_metadata_file, 'r') as f:
            for line in f:
                meta = json.loads(line)
                request_metadata_list[meta['sample_idx']] = meta
        print(f"Loaded metadata for {len(request_metadata_list)} requests")
    else:
        print(f"Warning: Request metadata file not found, will reconstruct from samples")
    
    # Fetch results
    print(f"\nFetching results from API...")
    results = []
    
    for result in client.messages.batches.results(batch_id):
        results.append(result)
    
    print(f"Retrieved {len(results)} results")
    
    # Process results
    print(f"\nProcessing results...")
    processed_results = []
    filtered_samples = []
    
    stats = {
        'total': len(results),
        'solvable_and_correct': 0,
        'solvable_but_wrong': 0,
        'not_solvable': 0,
        'parse_error': 0,
        'api_error': 0
    }
    
    failure_cases = []
    
    for result in tqdm(results, desc="Analyzing responses"):
        custom_id = result.custom_id
        sample_idx = int(custom_id)
        
        if sample_idx >= len(original_samples):
            print(f"Warning: sample_idx {sample_idx} out of range")
            continue
        
        original_sample = original_samples[sample_idx]
        
        # Get metadata (either from file or reconstruct)
        if sample_idx in request_metadata_list:
            metadata = request_metadata_list[sample_idx]
        else:
            # Reconstruct metadata if not available
            _, metadata = prepare_batch_request(original_sample, sample_idx)
        
        correct_choice = metadata['correct_choice']
        
        # Handle API errors
        if result.result.type == "errored":
            stats['api_error'] += 1
            failure_cases.append({
                'sample_idx': sample_idx,
                'sample': original_sample,
                'error': 'API Error',
                'details': str(result.result.error)
            })
            continue
        
        # Parse response
        response_text = result.result.message.content[0].text
        solvable, answer = parse_json_response(response_text)
        
        # Check for parse errors
        if solvable is None:
            stats['parse_error'] += 1
            failure_cases.append({
                'sample_idx': sample_idx,
                'sample': original_sample,
                'error': 'Parse Error',
                'response': response_text
            })
            continue
        
        # Categorize result
        if not solvable:
            stats['not_solvable'] += 1
            failure_cases.append({
                'sample_idx': sample_idx,
                'sample': original_sample,
                'error': 'Not Solvable',
                'solvable': solvable,
                'answer': answer
            })
        elif answer == correct_choice:
            stats['solvable_and_correct'] += 1
            # Add to filtered dataset
            filtered_samples.append(original_sample)
        else:
            stats['solvable_but_wrong'] += 1
            failure_cases.append({
                'sample_idx': sample_idx,
                'sample': original_sample,
                'error': 'Wrong Answer',
                'solvable': solvable,
                'predicted': answer,
                'correct': correct_choice,
                'options': f"A={metadata['option_a']}, B={metadata['option_b']}"
            })
        
        processed_results.append({
            'sample_idx': sample_idx,
            'sample': original_sample,
            'solvable': solvable,
            'answer': answer,
            'correct': answer == correct_choice if solvable else False
        })
    
    return {
        'stats': stats,
        'filtered_samples': filtered_samples,
        'failure_cases': failure_cases,
        'processed_results': processed_results,
        'batch_id': batch_id
    }


def save_filtered_dataset(filtered_samples: List[Dict], output_path: Path, stats: Dict):
    """Save filtered dataset to file."""
    print(f"\n{'='*60}")
    print(f"Saving filtered dataset")
    print(f"{'='*60}")
    
    with open(output_path, 'w') as f:
        for sample in filtered_samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Saved {len(filtered_samples)} samples to {output_path}")
    
    # Save statistics report
    stats_file = output_path.parent / f"{output_path.stem}_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Saved statistics to {stats_file}")


def print_statistics(stats: Dict, failure_cases: List[Dict], show_all_failures: bool = False, output_dir: Path = None):
    """Print detailed statistics and failure cases."""
    total = stats['total']
    
    # Save parse errors to file for detailed inspection
    if output_dir:
        parse_errors = [f for f in failure_cases if f.get('error') == 'Parse Error']
        if parse_errors:
            parse_error_file = output_dir / f"parse_errors_{int(time.time())}.jsonl"
            with open(parse_error_file, 'w') as f:
                for error in parse_errors:
                    f.write(json.dumps(error, indent=2) + '\n')
            print(f"Saved {len(parse_errors)} parse errors to {parse_error_file}")
    
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples: {total}")
    print(f"Solvable and correct: {stats['solvable_and_correct']} ({stats['solvable_and_correct']/total*100:.1f}%)")
    print(f"Solvable but wrong: {stats['solvable_but_wrong']} ({stats['solvable_but_wrong']/total*100:.1f}%)")
    print(f"Not solvable: {stats['not_solvable']} ({stats['not_solvable']/total*100:.1f}%)")
    print(f"Parse errors: {stats['parse_error']} ({stats['parse_error']/total*100:.1f}%)")
    print(f"API errors: {stats['api_error']} ({stats['api_error']/total*100:.1f}%)")
    
    # Success metrics
    solvable_total = stats['solvable_and_correct'] + stats['solvable_but_wrong']
    if solvable_total > 0:
        accuracy_on_solvable = stats['solvable_and_correct'] / solvable_total * 100
        print(f"\nAccuracy on solvable samples: {accuracy_on_solvable:.1f}%")
    
    # Show failure cases
    if failure_cases:
        max_failures = len(failure_cases) if show_all_failures else min(10, len(failure_cases))
        
        print(f"\n{'='*60}")
        print(f"Failure Cases (showing {max_failures} of {len(failure_cases)})")
        print(f"{'='*60}\n")
        
        for i, failure in enumerate(failure_cases[:max_failures]):
            sample = failure['sample']
            print(f"--- Failure #{i+1} (Sample #{failure['sample_idx']}) ---")
            print(f"Type: {failure['error']}")
            print(f"Entity: {sample['entity']}")
            print(f"Expected state: {sample['positive_state']}")
            
            if 'predicted' in failure:
                print(f"Predicted: {failure['predicted']}")
                print(f"Options: {failure.get('options', 'N/A')}")
            elif 'response' in failure:
                response = failure['response']
                print(f"Raw response ({len(response)} chars):")
                print(response)  # Show full response to diagnose truncation
            elif 'details' in failure:
                print(f"Error: {failure['details']}")
            
            print(f"\nContext (first 500 chars):")
            print(sample['context'][:500] + "..." if len(sample['context']) > 500 else sample['context'])
            print()


def main():
    """Main validation routine."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate and filter SelfIE dataset with Claude Haiku via Batch API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit test batch of 100 samples
  python validate_selfie_dataset.py --mode submit --num-samples 100 --split dev
  
  # Submit full dataset
  python validate_selfie_dataset.py --mode submit --split dev
  
  # Fetch results and create filtered dataset
  python validate_selfie_dataset.py --mode fetch --batch-id msgbatch_abc123
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['submit', 'fetch'],
        help='Mode: submit batch or fetch results'
    )
    parser.add_argument(
        '--split',
        type=str,
        default=None,
        choices=['train', 'dev'],
        help='Which split to process (required for submit mode)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Number of samples to test (default: all samples in split)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Max samples per batch (for splitting large datasets, e.g., 3000)'
    )
    parser.add_argument(
        '--start-idx',
        type=int,
        default=0,
        help='Starting index for batch (for manual splitting)'
    )
    parser.add_argument(
        '--batch-id',
        type=str,
        default=None,
        help='Batch ID to fetch (required for fetch mode)'
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
        '--show-all-failures',
        action='store_true',
        help='Show all failure cases (default: first 10)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/workspace/state-probes/data/batch_results',
        help='Directory for batch files and results'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'submit' and not args.split:
        parser.error("--split is required for submit mode")
    if args.mode == 'fetch' and not args.batch_id:
        parser.error("--batch-id is required for fetch mode")
    
    # Set random seed
    random.seed(args.seed)
    
    # Setup paths
    data_dir = Path("/workspace/state-probes/data/selfie_format")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Anthropic client
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found in environment")
        return
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # SUBMIT MODE
    if args.mode == 'submit':
        jsonl_path = data_dir / f"{args.split}.jsonl"
        
        if not jsonl_path.exists():
            print(f"Error: Dataset file not found: {jsonl_path}")
            return
        
        # Load samples
        print(f"\n{'='*60}")
        print(f"Loading samples from {args.split} split")
        print(f"{'='*60}")
        
        # Load all samples first
        all_samples = load_samples(str(jsonl_path), num_samples=None, property_filter=None)
        
        # Apply start_idx and batch_size if specified
        if args.start_idx > 0 or args.batch_size:
            start = args.start_idx
            end = start + args.batch_size if args.batch_size else len(all_samples)
            samples = all_samples[start:end]
            print(f"Loaded samples {start} to {end-1} ({len(samples)} samples)")
        elif args.num_samples:
            samples = all_samples[:args.num_samples] if args.num_samples else all_samples
            print(f"Loaded {len(samples)} samples")
            print(f"(First {args.num_samples} from dataset)")
        else:
            samples = all_samples
            print(f"Loaded {len(samples)} samples")
            print(f"(Full dataset)")
        
        # Submit batch
        batch_id = submit_batch(client, samples, output_dir, model=args.model, split=args.split)
        
        print(f"\n{'='*60}")
        print(f"Batch submission complete!")
        print(f"{'='*60}")
        print(f"\nBatch will be processed asynchronously by Anthropic.")
        print(f"Check back in a few hours.")
        
    # FETCH MODE
    elif args.mode == 'fetch':
        results = fetch_batch_results(client, args.batch_id, output_dir)
        
        if results is None:
            return
        
        stats = results['stats']
        filtered_samples = results['filtered_samples']
        failure_cases = results['failure_cases']
        
        # Print statistics
        print_statistics(stats, failure_cases, args.show_all_failures, output_dir)
        
        # Save filtered dataset
        # Determine output filename based on batch metadata
        metadata_file = output_dir / f"batch_metadata_{args.batch_id}.json"
        split_name = 'unknown'
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                # Get split name from metadata
                split_name = metadata.get('split', 'unknown') or 'unknown'
        
        # Save with batch_id in filename to avoid overwriting
        output_path = output_dir / f"{split_name}_filtered_{args.batch_id}.jsonl"
        save_filtered_dataset(filtered_samples, output_path, stats)
        
        # Also save a summary for easy combining later
        summary_file = output_dir / f"batch_summary_{args.batch_id}.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'batch_id': args.batch_id,
                'split': split_name,
                'filtered_file': str(output_path),
                'stats': stats
            }, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"FILTERING COMPLETE")
        print(f"{'='*60}")
        print(f"Original samples: {stats['total']}")
        print(f"Filtered samples: {len(filtered_samples)}")
        print(f"Retention rate: {len(filtered_samples)/stats['total']*100:.1f}%")
        print(f"\nFiltered dataset saved to: {output_path}")


if __name__ == '__main__':
    main()

