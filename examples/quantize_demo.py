# examples/quantize_demo.py
import torch
from deep_temporal_transformer.models.model_enhanced import DeepTemporalTransformer
from deep_temporal_transformer.utils.seeds import set_seed

def quantize_demo():
    set_seed(42)
    model = DeepTemporalTransformer(input_dim=128).eval()
    # Simple eager-mode static quantization example (PyTorch)
    model.qconfig = torch.quantization.default_qconfig
    torch.quantization.prepare(model, inplace=True)
    # Calibration: run a few batches through model (use real small dataset)
    # for batch in calib_loader: model(batch)
    torch.quantization.convert(model, inplace=True)
    torch.save(model.state_dict(), "models/dtt_quantized.pth")
    print("Quantized model saved at models/dtt_quantized.pth")

if __name__ == "__main__":
    quantize_demo()
