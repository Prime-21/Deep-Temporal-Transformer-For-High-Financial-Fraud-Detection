# bench/latency.py
import time, torch, argparse
from deep_temporal_transformer.models.model_enhanced import DeepTemporalTransformer

def measure(model, device, dummy_input, iters=1000, warm=100):
    model.to(device).eval()
    with torch.no_grad():
        for _ in range(warm):
            _ = model(dummy_input)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            _ = model(dummy_input)
        torch.cuda.synchronize()
    avg = (time.time() - t0) / iters
    print(f"avg forward-pass time (s): {avg:.8f}")
    return avg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--feat", type=int, default=128)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = DeepTemporalTransformer(input_dim=args.feat)
    dummy = torch.randn(args.batch, args.seq_len, args.feat).to(device)
    measure(model, device, dummy, iters=1000, warm=200)
