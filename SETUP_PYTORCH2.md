# Setup Guide for PyTorch 2.x

This guide shows how to get the state-probes project working with modern PyTorch 2.x (tested with PyTorch 2.9.1).

## Changes from Original

The main changes to support PyTorch 2.x:

1. **Updated dependencies** in `requirements.txt`:
   - PyTorch: 1.7.0 → 2.9.1+
   - Transformers: 4.4.2 → 4.57.1+
   - TextWorld: 1.4.0 → 1.6.2+
   - NumPy, tqdm, and other packages updated to compatible versions

2. **Fixed AdamW imports**: `AdamW` was moved from `transformers` to `torch.optim` in newer versions
   - Updated in: `probe_models.py`, `scripts/train_*.py`, `scripts/probe_*.py`

3. **Fixed syntax warning** in `data/textworld/tw_dataloader.py` (escape sequences)

## Installation

### Prerequisites
- Python 3.9+ (tested with 3.12)
- For better performance, use a fast filesystem for the virtual environment (not network-mounted)

### Quick Setup

```bash
# Create and activate virtual environment (use fast filesystem, not network-mounted)
python3 -m venv /tmp/state-probes-venv
source /tmp/state-probes-venv/bin/activate

# Install PyTorch 2.x (CPU version for maximum compatibility)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
cd /workspace/state-probes
pip install -r requirements.txt
```

### Environment Variables

Before running any commands, set these environment variables:

```bash
export PYTHONPATH=/workspace/state-probes
export TOKENIZERS_PARALLELISM=true
```

## TextWorld Workflow

### 1. Download TextWorld Data

```bash
wget http://web.mit.edu/bzl/www/tw_data.tar.gz
tar -xzvf tw_data.tar.gz
```

### 2. Train a BART or T5 Model on TextWorld Data

```bash
python scripts/train_textworld.py \
    --arch [t5|bart] [--no_pretrain] \
    --data tw_data/simple_traces --gamefile tw_games/simple_games
```

Model checkpoints are saved under `twModels/*`.

### 3. Get All TextWorld Proposition Encodings

```bash
python scripts/get_all_tw_facts.py \
    --data tw_data/simple_traces --gamefile tw_data/simple_games \
    --state_model_arch [bart|t5] \
    --probe_target belief_facts_pair \
    --state_model_path [None|pretrain|<path_to_lm_checkpoint>] \
    --out_file <path_to_prop_encodings>
```

### 4. Run the Probe

```bash
python scripts/probe_textworld.py \
    --arch [bart|t5] --data tw_data/simple_traces --gamefile tw_data/simple_games \
    --probe_target final.full_belief_facts_pair --encode_tgt_state NL.[bart|t5] \
    --localizer_type belief_facts_pair_[first|last|all] --probe_type 3linear_classify \
    --probe_agg_method avg --tgt_agg_method avg \
    --lm_save_path <path_to_lm_checkpoint> [--no_pretrain] \
    --ents_to_states_file <path_to_prop_encodings> \
    --eval_batchsize 256 --batchsize 32
```

For evaluation, add `--eval_only --probe_save_path <path_to_probe_checkpoint>`.

Probe checkpoints are saved under `probe_models_textworld/*`.

### 5. Print Metrics

```bash
python scripts/print_metrics.py \
    --arch [bart|t5] --domain textworld \
    --pred_files <path_to_model_predictions_1>,<path_to_model_predictions_2>,... \
    [--use_remap_domain --remap_fn <path_to_remap_model_predictions>] \
    [--single_side_probe]
```

## Verification

Test that everything is working:

```bash
source /tmp/state-probes-venv/bin/activate
export PYTHONPATH=/workspace/state-probes
export TOKENIZERS_PARALLELISM=true

# Test imports
python -c "
import torch
import transformers
import textworld
from probe_models import get_lang_model
from data.textworld.tw_dataloader import TWDataset
print('✓ All imports successful!')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'TextWorld: {textworld.__version__}')
"
```

## Compatibility Notes

- The code has been updated to work with PyTorch 2.x and Transformers 4.x
- All original functionality should be preserved
- Some APIs in newer transformers versions have changed slightly, but backward compatibility is maintained
- If you encounter issues with specific model architectures, check the transformers documentation for API changes

## Troubleshooting

**Issue**: Import errors related to `AdamW`  
**Solution**: All `AdamW` imports have been updated to use `torch.optim.AdamW` instead of `transformers.AdamW`

**Issue**: TextWorld installation fails  
**Solution**: We use TextWorld 1.6.2+ which has pre-built wheels and doesn't require manual Inform7 setup

**Issue**: Out of memory errors  
**Solution**: Use CPU version of PyTorch or reduce batch sizes in training/probe scripts

