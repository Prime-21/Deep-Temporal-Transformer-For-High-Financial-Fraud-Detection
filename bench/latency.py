# bench/latency.py
"""
Benchmark script for model-only forward pass and end-to-end measurement.
Usage:
  # model-only
  python bench/latency.py --mode model_only --device cuda:0 --iters 1000

  # end-to-end (includes a small synthetic preprocessing step)
  python bench/latency.py --mode end2end --device cuda:0 --iters 500
"""
import time
import argparse
import torch
from deep_temporal_transformer.models.model_enhanced import DeepTemporalTransformer
from deep_temporal_transformer.data.data import DataProcessor

def measure_model_only(model, device, dummy, iters=1000, warm=200):
    model.to(device).eval()
    with torch.no_grad():
        for _ in range(warm):
            _ = model(dummy)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            _ = model(dummy)
        torch.cuda.synchronize()
    avg = (time.time() - t0) / iters
    print(f"avg forward-pass time (s): {avg:.8f}")
    print(f"avg forward-pass time (ms): {avg*1000:.6f}")
    return avg

def measure_end2end(model, device, processor, iters=500, warm=50):
    model.to(device).eval()
    # Prepare synthetic batched input via DataProcessor to simulate full pipeline
    with torch.no_grad():
        dummy = processor.get_dummy_tensor(batch_size=1).to(device)
        # Warm-up
        for _ in range(warm):
            _ = model(dummy)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            # include a small synthetic preprocessing step
            pre = processor.preprocess(dummy)
            out = model(pre)
            _ = out.cpu().numpy()
        torch.cuda.synchronize()
    avg = (time.time() - t0) / iters
    print(f"avg end-to-end time (s): {avg:.8f}")
    print(f"avg end-to-end time (ms): {avg*1000:.6f}")
    return avg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=["model_only", "end2end"], default="model_only")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--warm", type=int, default=200)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # NOTE: adjust input_dim / seq_len to match your trained config
    config_input_dim = 128
    seq_len = 8
    model = DeepTemporalTransformer(input_dim=config_input_dim)
    dummy = torch.randn(1, seq_len, config_input_dim).to(device)

    # optional DataProcessor usage for end2end
    from deep_temporal_transformer.data.data import DataProcessor
    processor = DataProcessor(seq_len=seq_len)
    if args.mode == "model_only":
        measure_model_only(model, device, dummy, iters=args.iters, warm=args.warm)
    else:
        measure_end2end(model, device, processor, iters=int(args.iters/2), warm=50)
