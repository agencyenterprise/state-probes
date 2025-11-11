# TextWorld Quick Start Guide

This guide focuses on getting the TextWorld experiments running with PyTorch 2.x.

## 🚀 Setup (First Time Only)

### 1. Install Dependencies

The virtual environment is already created at `/tmp/state-probes-venv`. Install dependencies:

```bash
cd /workspace/state-probes
bash setup_venv.sh
```

This will:
- Install PyTorch 2.9.1+ (CPU version)
- Install Transformers 4.57.1+
- Install TextWorld 1.6.2+
- Install all other dependencies

### 2. Verify Installation

```bash
source /tmp/state-probes-venv/bin/activate
export PYTHONPATH=/workspace/state-probes
export TOKENIZERS_PARALLELISM=true
python test_setup.py
```

You should see: `✓ All tests passed! Setup is working correctly.`

## 🎯 Running TextWorld Experiments

### Quick Environment Setup

For each new terminal session, run:

```bash
source /workspace/state-probes/activate_env.sh
```

This activates the virtual environment and sets required environment variables.

### Step 1: Download TextWorld Data

```bash
# Download and extract TextWorld data
wget http://web.mit.edu/bzl/www/tw_data.tar.gz
tar -xzvf tw_data.tar.gz
```

The data will be in `tw_data/` directory with structure:
- `tw_data/simple_traces/` - game traces
- `tw_data/simple_games/` - game files

### Step 2: Train Language Model (BART or T5)

Train a BART model:

```bash
python scripts/train_textworld.py \
    --arch bart \
    --data tw_data/simple_traces \
    --gamefile tw_data/simple_games \
    --epochs 100 \
    --batchsize 16 \
    --lr 1e-5
```

Or train a T5 model:

```bash
python scripts/train_textworld.py \
    --arch t5 \
    --data tw_data/simple_traces \
    --gamefile tw_data/simple_games \
    --epochs 100 \
    --batchsize 16 \
    --lr 1e-5
```

**Options:**
- `--no_pretrain`: Train from scratch (not from pretrained checkpoint)
- `--save_path PATH`: Specify custom save path
- `--device cuda`: Use GPU (if available)

**Output:** Model saved to `twModels/` directory

### Step 3: Generate Proposition Encodings

Before training probes, generate encodings for all possible propositions:

```bash
python scripts/get_all_tw_facts.py \
    --data tw_data/simple_traces \
    --gamefile tw_data/simple_games \
    --state_model_arch bart \
    --probe_target belief_facts_pair \
    --state_model_path twModels/your_model.p \
    --out_file tw_prop_encodings.pkl
```

**Key parameters:**
- `--state_model_arch`: Use same arch as training (bart or t5)
- `--state_model_path`: Path to trained model, or "pretrain" for base model, or "None" for control
- `--probe_target`: Type of facts to encode
  - `belief_facts_pair`: Entity pairs
  - `belief_facts_single`: Single entities

**Output:** Encodings saved to specified file (e.g., `tw_prop_encodings.pkl`)

### Step 4: Train Probe

Train a probe to decode world state from model representations:

```bash
python scripts/probe_textworld.py \
    --arch bart \
    --data tw_data/simple_traces \
    --gamefile tw_data/simple_games \
    --probe_target final.full_belief_facts_pair \
    --encode_tgt_state NL.bart \
    --localizer_type belief_facts_pair_all \
    --probe_type 3linear_classify \
    --probe_agg_method avg \
    --tgt_agg_method avg \
    --lm_save_path twModels/your_model.p \
    --ents_to_states_file tw_prop_encodings.pkl \
    --batchsize 32 \
    --eval_batchsize 256 \
    --lr 1e-4 \
    --epochs 50
```

**Key parameters:**

**Probe target** (`--probe_target`):
- `init.full_belief_facts_pair`: Decode initial state
- `final.full_belief_facts_pair`: Decode final state
- `*.control_with_rooms`: For remap experiments

**Localizer** (`--localizer_type`):
- `belief_facts_pair_all`: Use all tokens
- `belief_facts_pair_first`: Use first mention
- `belief_facts_pair_last`: Use last mention

**Probe type** (`--probe_type`):
- `3linear_classify`: 3-way classification (used for proposition classification)
- `linear_classify`: Binary classification
- `linear_retrieve`: Retrieval-based
- `decoder`: Full decoder model

**Output:** Probe saved to `probe_models_textworld/` directory

### Step 5: Evaluate Probe

Evaluate a trained probe:

```bash
python scripts/probe_textworld.py \
    --eval_only \
    --probe_save_path probe_models_textworld/your_probe.p \
    --arch bart \
    --data tw_data/simple_traces \
    --gamefile tw_data/simple_games \
    --probe_target final.full_belief_facts_pair \
    --encode_tgt_state NL.bart \
    --localizer_type belief_facts_pair_all \
    --probe_type 3linear_classify \
    --probe_agg_method avg \
    --tgt_agg_method avg \
    --lm_save_path twModels/your_model.p \
    --ents_to_states_file tw_prop_encodings.pkl \
    --eval_batchsize 256
```

**Output:** Predictions saved to `.jsonl` file in same directory as probe

### Step 6: Print Metrics

Calculate detailed metrics from predictions:

```bash
python scripts/print_metrics.py \
    --arch bart \
    --domain textworld \
    --pred_files probe_models_textworld/predictions_1.jsonl,probe_models_textworld/predictions_2.jsonl
```

**Options:**
- `--use_remap_domain`: For remap experiments
- `--remap_fn PATH`: Path to remap model predictions
- `--single_side_probe`: If using single-entity probes

**Output:** Prints detailed metrics including:
- Entity exact match (EM)
- State exact match
- Relation vs. proposition accuracy
- Per-relation breakdowns

## 🔬 Experiment Variations

### No Language Model (Control)

Test without using the language model:

```bash
python scripts/probe_textworld.py \
    --control_input \
    [... other parameters ...]
```

### Decode Initial State

Instead of final state, decode the initial state:

```bash
python scripts/probe_textworld.py \
    --probe_target init.full_belief_facts_pair \
    [... other parameters ...]
```

### Single Entity Probes

Decode single entities instead of pairs:

1. Generate single-entity encodings:
```bash
python scripts/get_all_tw_facts.py \
    --probe_target belief_facts_single \
    [... other parameters ...]
```

2. Train probe:
```bash
python scripts/probe_textworld.py \
    --probe_target final.full_belief_facts_single \
    --localizer_type belief_facts_single_all \
    [... other parameters ...]
```

### Remap Experiments

Test generalization with remapped room names:

```bash
python scripts/probe_textworld.py \
    --probe_target final.full_belief_facts_pair.control_with_rooms \
    [... other parameters ...]
```

## 📊 Understanding Output

### Model Checkpoints
- Location: `twModels/`
- Format: PyTorch state dict (`.p` files)
- Contains: Full model weights

### Probe Checkpoints
- Location: `probe_models_textworld/`
- Format: PyTorch state dict (`.p` files)
- Contains: Probe model weights only

### Predictions
- Location: Same directory as probe checkpoint
- Format: JSONL (one JSON per line)
- Contains: Ground truth and predicted states per example

### Metrics
- Printed to stdout
- Includes: EM, accuracy, per-relation stats

## 💡 Tips

1. **Start small**: Use `--train_data_size 100` to test with small dataset first
2. **Monitor training**: Watch loss and EM metrics during training
3. **Batch sizes**: Reduce if you run out of memory
4. **Learning rate**: Default 1e-4 works well for probes, 1e-5 for LMs
5. **Patience**: LM training can take several hours; probe training is faster

## 🐛 Common Issues

**Issue:** "FileNotFoundError: tw_data/simple_traces"  
**Solution:** Make sure you downloaded and extracted the TextWorld data

**Issue:** "RuntimeError: CUDA out of memory"  
**Solution:** Use `--device cpu` or reduce batch sizes

**Issue:** "No module named 'probe_models'"  
**Solution:** Make sure `PYTHONPATH=/workspace/state-probes` is set

**Issue:** Probe accuracy is very low  
**Solution:** 
- Check that you're using the correct language model checkpoint
- Verify proposition encodings were generated correctly
- Try training longer or with different hyperparameters

## 📖 Further Reading

See `MIGRATION_SUMMARY.md` for details on PyTorch 2.x changes.
See `SETUP_PYTORCH2.md` for detailed setup instructions.
See `README.md` for original documentation.

## 🎉 You're Ready!

You now have a fully working setup for running TextWorld experiments with PyTorch 2.x!

