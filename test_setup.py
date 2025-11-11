#!/usr/bin/env python
"""
Quick test to verify PyTorch 2.x compatibility
"""
import sys

print("Testing PyTorch 2.x compatibility...")
print("=" * 60)

# Test 1: Basic imports
print("\n1. Testing basic imports...")
try:
    import torch
    import transformers
    import textworld
    import numpy as np
    import tqdm
    print(f"   ✓ PyTorch: {torch.__version__}")
    print(f"   ✓ Transformers: {transformers.__version__}")
    print(f"   ✓ TextWorld: {textworld.__version__}")
    print(f"   ✓ NumPy: {np.__version__}")
except ImportError as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

# Test 2: Project imports
print("\n2. Testing project-specific imports...")
try:
    from probe_models import (
        get_lang_model, get_state_encoder, get_probe_model,
        ProbeLinearModel, ProbeConditionalGenerationModel
    )
    from data.textworld.tw_dataloader import TWDataset, TWFullDataLoader
    from localizer.tw_localizer import TWLocalizer
    from metrics.tw_metrics import get_em
    print("   ✓ probe_models")
    print("   ✓ data.textworld.tw_dataloader")
    print("   ✓ localizer.tw_localizer")
    print("   ✓ metrics.tw_metrics")
except ImportError as e:
    print(f"   ✗ Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: AdamW import (critical for PyTorch 2.x compatibility)
print("\n3. Testing AdamW import...")
try:
    from torch.optim import AdamW
    print("   ✓ AdamW imported from torch.optim")
except ImportError as e:
    print(f"   ✗ AdamW import error: {e}")
    sys.exit(1)

# Test 4: Model loading test
print("\n4. Testing model initialization...")
try:
    from transformers import BartTokenizerFast, T5TokenizerFast
    
    # Test BART tokenizer
    bart_tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')
    print("   ✓ BART tokenizer loaded")
    
    # Test T5 tokenizer  
    t5_tokenizer = T5TokenizerFast.from_pretrained('t5-base')
    print("   ✓ T5 tokenizer loaded")
    
    # Quick encoding test
    test_text = "This is a test."
    bart_encoded = bart_tokenizer(test_text, return_tensors='pt')
    t5_encoded = t5_tokenizer(test_text, return_tensors='pt')
    print("   ✓ Tokenizers work correctly")
    
except Exception as e:
    print(f"   ✗ Model initialization error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Check CUDA availability
print("\n5. Checking device availability...")
if torch.cuda.is_available():
    print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
else:
    print("   ℹ CUDA not available (CPU-only mode)")

print("\n" + "=" * 60)
print("✓ All tests passed! Setup is working correctly.")
print("=" * 60)

