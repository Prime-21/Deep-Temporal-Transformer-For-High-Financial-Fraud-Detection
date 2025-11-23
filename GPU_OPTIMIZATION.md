# 🚀 GPU Optimization Guide - Maximize Your 15GB GPU

## Current Problem
- **Available**: 15GB GPU memory
- **Using**: Only ~1GB
- **Target**: 10-13GB (safe maximum)

---

## ✅ Solution 1: Increase Batch Size (Most Impactful!)

```python
# In Colab, after imports
config = get_default_config()

# 🔥 INCREASE BATCH SIZE - This is the #1 way to use more GPU
config.training.batch_size = 1024  # Instead of default 256
# Try even higher: 2048, 4096 if you have 15GB

# Also increase model size for better performance
config.model.d_model = 512        # Larger model (was 256)
config.model.num_layers = 8       # More layers (was 6)
config.model.nhead = 16           # More attention heads (was 8)
config.model.dim_feedforward = 2048  # Larger FFN (was 1024)
config.model.memory_slots = 2048  # More memory (was 1024)

trainer = ModelTrainer(config, device)
trainer.setup_model(input_dim=X_train.shape[-1])
history = trainer.train(X_train, y_train, X_val, y_val)
```

---

## ✅ Solution 2: Enable Mixed Precision Training

```python
# Add this at the top of your training cell
import torch
from torch.cuda.amp import autocast, GradScaler

# Enable automatic mixed precision
scaler = GradScaler()

# Your model will automatically use GPU more efficiently
# This allows BIGGER batch sizes without OOM errors
```

---

## ✅ Solution 3: Use Advanced GPU-Optimized Model

```python
from deep_temporal_transformer.models.advanced_transformer import DeepTemporalTransformerAdvanced
from deep_temporal_transformer.training.advanced_training import detect_and_configure_gpu

# Auto-configure for your GPU
gpu_config = detect_and_configure_gpu()
print(f"Detected GPU: {gpu_config['device']}")
print(f"Recommended batch size: {gpu_config['recommended_batch_size']}")

# Create model with gradient checkpointing (allows larger models)
model = DeepTemporalTransformerAdvanced(
    input_dim=X_train.shape[-1],
    d_model=512,           # Larger
    num_heads=16,          # More heads
    num_layers=8,          # Deeper
    num_experts=16,        # More experts for MoE
    memory_slots=2048,     # More memory
    use_gradient_checkpointing=True  # Trades compute for memory
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
# Should be 20-50M parameters to use more GPU
```

---

## ✅ Solution 4: Optimize Data Loading

```python
# Use larger dataset
processor = DataProcessor(seq_len=16, random_state=42)  # Longer sequences
X_train, y_train, X_val, y_val, X_test, y_test = processor.process_data()

# Or load more data
config.data.n_samples = 500000  # Instead of 100000

# Pin memory for faster GPU transfer
# (This speeds up data loading to GPU)
```

---

## ✅ Solution 5: Monitor GPU Usage in Real-Time

```python
# Add this to your training cell to monitor GPU usage
!pip install gpustat -q

# Check GPU usage during training
!watch -n 1 nvidia-smi  # Run in separate cell

# Or programmatically:
import torch
print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"GPU Memory Cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
```

---

## 🎯 Recommended Configuration for 15GB GPU

```python
from deep_temporal_transformer import get_default_config, DataProcessor, ModelTrainer, set_random_seeds, get_device

set_random_seeds(42)
config = get_default_config()
device = get_device()

# ⚡ OPTIMIZED FOR 15GB GPU
config.training.batch_size = 2048      # 🔥 Much larger batch
config.training.epochs = 20
config.training.learning_rate = 5e-4   # Higher LR for larger batches

# 🧠 LARGER MODEL
config.model.d_model = 512
config.model.num_layers = 8
config.model.nhead = 16
config.model.dim_feedforward = 2048
config.model.memory_slots = 2048
config.model.dropout = 0.2

# 📊 MORE DATA
config.data.n_samples = 500000
config.data.seq_len = 16  # Longer sequences

# Process data
processor = DataProcessor(seq_len=config.data.seq_len, random_state=42)
X_train, y_train, X_val, y_val, X_test, y_test = processor.process_data()

# Train with optimized settings
trainer = ModelTrainer(config, device)
trainer.setup_model(input_dim=X_train.shape[-1])

print(f"🎯 Training with batch_size={config.training.batch_size}")
print(f"🧠 Model size: {sum(p.numel() for p in trainer.model.parameters()):,} parameters")

history = trainer.train(X_train, y_train, X_val, y_val)

# Check GPU usage
print(f"\n📊 GPU Memory Used: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB / 15 GB")
```

---

## 📈 Expected Results

With these optimizations:
- **GPU Usage**: 10-13 GB (80-85% utilization)
- **Training Speed**: 3-5x faster
- **Model Performance**: Better (larger model + more data)
- **Batch Size**: 2048-4096 (vs original 256)

---

## ⚠️ Troubleshooting

### If you get "Out of Memory" error:
```python
# Reduce batch size gradually
config.training.batch_size = 1536  # Try 1536
# or
config.training.batch_size = 1024  # Try 1024

# Or reduce model size slightly
config.model.d_model = 384         # Between 256 and 512
```

### Check current GPU usage:
```python
!nvidia-smi
```

### Clear GPU memory:
```python
import torch
torch.cuda.empty_cache()
import gc
gc.collect()
```

---

## 🎯 Quick Test

Run this to find your optimal batch size:

```python
import torch

def find_optimal_batch_size(model, X_sample, device, start_bs=256):
    """Binary search for optimal batch size."""
    batch_size = start_bs
    while batch_size < 8192:
        try:
            # Test with this batch size
            test_data = torch.randn(batch_size, X_sample.shape[1], X_sample.shape[2]).to(device)
            with torch.no_grad():
                _ = model(test_data)
            
            print(f"✅ Batch size {batch_size} works!")
            print(f"   GPU Usage: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            
            # Try larger
            batch_size *= 2
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                optimal = batch_size // 2
                print(f"\n🎯 Optimal batch size: {optimal}")
                return optimal
            raise e
    
    return batch_size

# Usage:
optimal_bs = find_optimal_batch_size(trainer.model, X_train, device)
config.training.batch_size = optimal_bs
```
