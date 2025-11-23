# 🚀 Google Colab Quick Start Guide

## ✅ VERIFIED WORKING STEPS

Follow these steps **exactly in order** to run your Deep Temporal Transformer in Google Colab Pro:

---

## 📋 Cell 1: Check GPU
```python
# Check GPU availability
!nvidia-smi
```

---

## 📋 Cell 2: Install Dependencies
```python
# Install all required packages
!pip install torch torchvision scikit-learn pandas matplotlib seaborn -q
print("✅ Dependencies installed!")
```

---

## 📋 Cell 3: Clone Repository
```python
# Clean start - remove any existing directory
!rm -rf Deep-Temporal-Transformer-For-High-Financial-Fraud-Detection

# Clone from GitHub
!git clone https://github.com/Prime-21/Deep-Temporal-Transformer-For-High-Financial-Fraud-Detection.git

# Navigate to directory
%cd Deep-Temporal-Transformer-For-High-Financial-Fraud-Detection

# Verify location
!pwd
```

---

## 📋 Cell 4: Apply Runtime Patches
```python
# ==================================================================
# PATCH: Fix any outdated class names in train.py
# ==================================================================
import os

train_file = 'deep_temporal_transformer/training/train.py'

# Read the file
with open(train_file, 'r') as f:
    content = f.read()

# Apply fixes
content = content.replace('DeepTemporalTransformer(', 'DeepTemporalTransformerEnhanced(')
content = content.replace('FocalLoss(', 'FocalLossEnhanced(')
content = content.replace('Optional[DeepTemporalTransformer]', 'Optional[DeepTemporalTransformerEnhanced]')
content = content.replace("'DeepTemporalTransformer'", "'DeepTemporalTransformerEnhanced'")

# Write back
with open(train_file, 'w') as f:
    f.write(content)

print("✅ Runtime patches applied!")
```

---

## 📋 Cell 5: Install Package
```python
# Install in editable mode
!pip install -e . -q
print("✅ Package installed!")
```

---

## 📋 Cell 6: Restart Runtime
```python
# IMPORTANT: Restart the runtime to clear cached modules
# Click: Runtime → Restart runtime
# Then continue with the cells below
print("⚠️  STOP! Click Runtime → Restart runtime now, then run the next cell")
```

---

## 📋 Cell 7: Import and Setup (Run AFTER restart)
```python
from deep_temporal_transformer import (
    get_default_config, DataProcessor, ModelTrainer,
    set_random_seeds, get_device
)

# Setup
set_random_seeds(42)
config = get_default_config()
device = get_device()

print(f"✅ Imports successful!")
print(f"🚀 Using device: {device}")
print(f"📊 Model config: d_model={config.model.d_model}, layers={config.model.num_layers}")
```

---

## 📋 Cell 8: Configure Training (GPU Optimized for 15GB)
```python
# ⚡ OPTIMIZED FOR 15GB GPU - Maximum Performance!
config.training.batch_size = 2048      # 🔥 8x larger (was 256)
config.training.epochs = 20            # More epochs for better training
config.training.learning_rate = 5e-4   # Higher LR for larger batches

# 🧠 LARGER MODEL (uses 10-12 GB of your 15GB GPU)
config.model.d_model = 512             # Bigger model (was 256)
config.model.num_layers = 8            # Deeper (was 6)
config.model.nhead = 16                # More attention heads (was 8)
config.model.dim_feedforward = 2048    # Larger FFN (was 1024)
config.model.memory_slots = 2048       # More memory (was 1024)

# 📊 MORE DATA
config.data.n_samples = 500000         # More training data
config.data.seq_len = 16               # Longer sequences (was 8)

print(f"✅ GPU-Optimized Config:")
print(f"   Batch Size: {config.training.batch_size}")
print(f"   Model Dimension: {config.model.d_model}")
print(f"   Expected GPU Usage: 10-12 GB / 15 GB")
```

---

## 📋 Cell 9: Load Data
```python
# Load and process data (using optimized sequence length)
processor = DataProcessor(seq_len=config.data.seq_len, random_state=42)
X_train, y_train, X_val, y_val, X_test, y_test = processor.process_data()

print(f"✅ Data loaded:")
print(f"  Train: {X_train.shape}")
print(f"  Val: {X_val.shape}")
print(f"  Test: {X_test.shape}")
print(f"  Fraud rate: {y_train.mean():.2%}")
```

---

## 📋 Cell 10: Train Model
```python
# Create trainer
trainer = ModelTrainer(config, device)
trainer.setup_model(input_dim=X_train.shape[-1])

print(f"🧠 Model initialized:")
print(f"   Parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
print(f"   Expected size: 20-50M parameters")
print("\n🎯 Starting training...")

# Train!
history = trainer.train(X_train, y_train, X_val, y_val)

# Check GPU usage
import torch
gpu_used = torch.cuda.max_memory_allocated() / 1024**3
print(f"\n📊 GPU Memory Used: {gpu_used:.2f} GB / 15 GB ({gpu_used/15*100:.1f}%)")
print("✅ Training complete!")
```

---

## 📋 Cell 11: Evaluate
```python
# Evaluate on test set
results = trainer.evaluate_model(X_test, y_test)

print("\n" + "="*60)
print("📊 TEST SET RESULTS")
print("="*60)
print(f"F1 Score:       {results['f1']:.4f}")
print(f"AUC Score:      {results['auc']:.4f}")
print(f"Precision:      {results['precision']:.4f}")
print(f"Recall:         {results['recall']:.4f}")
print(f"Inference Time: {results['avg_inference_time']:.6f}s")
print("="*60)
```

---

## 📋 Cell 11 (Optional): Visualizations
```python
from deep_temporal_transformer.evaluation.explain import ModelExplainer
import matplotlib.pyplot as plt

# Create explainer
explainer = ModelExplainer(trainer.model, device)

# Plot training history
explainer.plot_training_history(history)
plt.show()

# Plot confusion matrix
explainer.plot_confusion_matrix(X_test, y_test)
plt.show()

print("✅ Visualizations complete!")
```

---

## 🎉 Success!

You've successfully:
- ✅ Set up the environment  
- ✅ Trained the Deep Temporal Transformer  
- ✅ Evaluated performance  
- ✅ Generated visualizations  

**Next Steps:**
- Use your own dataset instead of synthetic data
- Experiment with hyperparameters
- Try the advanced model with uncertainty estimation
- Generate thesis plots and results

---

## ⚠️ Troubleshooting

**If you get "DeepTemporalTransformer not defined" error:**
1. Make sure you ran Cell 4 (patches)
2. Make sure you restarted runtime after Cell 6
3. Reimport in a fresh cell after restart

**If imports fail:**
- Check you're in the correct directory: `/content/Deep-Temporal-Transformer-For-High-Financial-Fraud-Detection`
- Verify with: `!pwd`

**If training fails:**
- Check GPU is available: `!nvidia-smi`
- Reduce batch size if out of memory: `config.training.batch_size = 128`
