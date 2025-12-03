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

