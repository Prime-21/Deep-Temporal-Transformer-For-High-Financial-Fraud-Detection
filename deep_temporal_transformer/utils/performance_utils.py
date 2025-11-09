"""Performance optimization utilities."""
import torch
import numpy as np
from typing import Tuple, Optional
import gc


def optimize_memory():
    """Clear GPU and system memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def efficient_batch_processing(data: np.ndarray, batch_size: int, device: torch.device):
    """Memory-efficient batch processing generator."""
    n_samples = len(data)
    for i in range(0, n_samples, batch_size):
        batch = data[i:i + batch_size]
        yield torch.tensor(batch, dtype=torch.float32, device=device, pin_memory=True)


@torch.jit.script
def fast_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Optimized sigmoid function."""
    return torch.sigmoid(x)


def validate_tensor_inputs(tensor: torch.Tensor, expected_shape: Optional[Tuple[int, ...]] = None) -> bool:
    """Fast tensor validation."""
    if not isinstance(tensor, torch.Tensor):
        return False
    if expected_shape and tensor.shape != expected_shape:
        return False
    return torch.isfinite(tensor).all().item()