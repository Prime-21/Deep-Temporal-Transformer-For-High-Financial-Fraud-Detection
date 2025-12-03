# Deep Temporal Transformer (DTT) for High-Frequency Financial Fraud Detection

A high-performance, **real-time fraud detection framework** built using a hybrid **Deep Temporal Transformer (DTT)** architecture with multi-scale attention, temporal encoding, memory augmentation, and compliance-ready interpretability.

This repository contains the **complete implementation**, **benchmark scripts**, **preprocessing pipelines**, and **evaluation framework** used in the associated thesis.

---

## 🚀 Key Capabilities

### 🧠 Deep Temporal Transformer Architecture
- Multi-scale temporal attention  
- Memory-augmented pattern retrieval  
- Temporal positional encoding  
- Unified feature fusion for heterogeneous financial data  

### ⚡ Real-Time Performance
Two latency values are reported and benchmarked using `bench/latency.py`:
- **Model-only forward pass:** 0.003 ms  
- **End-to-end latency (preprocessing → inference → decoding):** 0.8–1.3 ms  

### 🎯 Robust Fraud Detection
- F1-Score: **0.8567**  
- Recall: **0.8912**  
- AUC-ROC: **0.9234**  
- Handles fraud rates below 0.1% via focal loss + CTGAN augmentation  

### 🔍 Full Explainability Stack
- SHAP value explanations  
- Attention heatmaps  
- Natural-language rationales  
- Compliance-aligned audit outputs  

### 🛡 Production-Ready Features
- Input validation  
- Security filters (SQLi / path traversal safeguards)  
- Mixed precision (FP16)  
- High-throughput GPU inference  

---

## 📁 Repository Structure

deep_temporal_transformer/
├── models/ # Transformer + Memory architectures
├── data/ # Preprocessing + CTGAN augmentation
├── utils/ # Seeds, metrics, helpers
├── training/ # Training pipelines
├── evaluation/ # Explainability tools
├── examples/ # Runnable demos (main, demo, quantization)
├── bench/ # Latency benchmarking scripts + logs
├── tests/ # Basic unit tests
├── configs/ # Default model/training configs
├── COLAB_QUICK_START.md
├── GPU_OPTIMIZATION.md
└── README.md

---

## 🧩 Installation

```bash
git clone https://github.com/Prime-21/Deep-Temporal-Transformer-For-High-Financial-Fraud-Detection
cd Deep-Temporal-Transformer-For-High-Financial-Fraud-Detection

pip install -e .
python validate_codebase.py

GPU acceleration requires a CUDA-enabled PyTorch installation.

🚀 Quick Start Example

from deep_temporal_transformer import (
    get_default_config, DataProcessor, ModelTrainer,
    set_random_seeds, get_device
)

set_random_seeds(42)
config = get_default_config()
device = get_device()

processor = DataProcessor(seq_len=8)
X_train, y_train, X_val, y_val, X_test, y_test = processor.process_data()

trainer = ModelTrainer(config, device)
trainer.setup_model(input_dim=X_train.shape[-1])

trainer.train(X_train, y_train, X_val, y_val)
results = trainer.evaluate_model(X_test, y_test)

print("F1:", results["f1"])
print("AUC:", results["auc"])
print("Latency:", results["avg_inference_time"])
```

⚡ Benchmarking Latency

```
python bench/latency.py --mode model_only
python bench/latency.py --mode end2end
```

Benchmark logs are stored automatically in:

```
bench/logs/
```

📊 Model Performance Summary

| Model                         | F1         | AUC        | Precision  | Recall     | Latency                                             |
| ----------------------------- | ---------- | ---------- | ---------- | ---------- | --------------------------------------------------- |
| Random Forest                 | 0.7234     | 0.8456     | 0.6891     | 0.7623     | 0.2 ms                                              |
| Logistic Regression           | 0.6789     | 0.8123     | 0.6234     | 0.7456     | 0.1 ms                                              |
| **Deep Temporal Transformer** | **0.8567** | **0.9234** | **0.8234** | **0.8912** | **0.003 ms (model-only) / 0.8–1.3 ms (end-to-end)** |

### 🔍 Explainability Tools

- SHAP feature attribution
- Attention heatmaps
- Decision-path tracing
- Natural-language decision rationale generation

These tools support:
- GDPR Article 22
- PSD2 risk-based authentication
- ECOA/FCRA review workflows

🧪 Testing
```
pytest -q
python validate_codebase.py
```

🏗 Deployment

- Docker-ready configuration
- Low-latency inference path
- Quantization (examples/quantize_demo.py)

GPU optimization notes in GPU_OPTIMIZATION.md

🧵 Citation
```
@thesis{dtt_fraud_detection_2025,
  title={Deep Temporal Transformer for High-Frequency Financial Fraud Detection},
  author={Prasad Kharat},
  year={2025},
  institution={University / Institute},
}
```

### ⭐ Acknowledgements

- NVIDIA A100 GPU provided for benchmarking
- IEEE-CIS Fraud dataset
- PyTorch, Scikit-Learn, CTGAN
- Supervisor & reviewers for guidance