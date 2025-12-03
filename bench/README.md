# Benchmarks and latency measurement

This folder contains scripts used to measure model-only and end-to-end latency.

Environment used for reported numbers:
- GPU: NVIDIA A100 (40GB)
- CUDA: 11.8
- PyTorch: 2.0
- Batch size: 1
- Sequence length: 8
- Warm-up iterations: 200
- Model compiled with CUDA Graphs where available

Model-only measurement:
- Runs `model.forward()` with preallocated CUDA tensors.
- Measures pure compute time (no I/O / serialization).

End-to-end measurement:
- Simulates a minimal preprocessing pipeline using `DataProcessor`,
  includes serialization and result decoding to approximate real deployment.

Run (example):
```bash
python bench/latency.py --mode model_only --device cuda:0 --iters 1000 --warm 200
python bench/latency.py --mode end2end --device cuda:0 --iters 500 --warm 50
