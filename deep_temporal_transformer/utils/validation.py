"""Input validation utilities."""
import numpy as np
import torch
from typing import Any, Union


def validate_array_input(data: np.ndarray, name: str = "data") -> None:
    """Validate numpy array input."""
    if not isinstance(data, np.ndarray):
        raise TypeError(f"{name} must be numpy array, got {type(data)}")
    
    if data.size == 0:
        raise ValueError(f"{name} cannot be empty")
    
    if not np.isfinite(data).all():
        raise ValueError(f"{name} contains non-finite values")


def validate_tensor_input(data: torch.Tensor, name: str = "tensor") -> None:
    """Validate tensor input."""
    if not isinstance(data, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor, got {type(data)}")
    
    if data.numel() == 0:
        raise ValueError(f"{name} cannot be empty")
    
    if not torch.isfinite(data).all():
        raise ValueError(f"{name} contains non-finite values")


def validate_labels(labels: np.ndarray) -> None:
    """Validate binary classification labels."""
    validate_array_input(labels, "labels")
    
    unique_labels = np.unique(labels)
    if not np.array_equal(unique_labels, [0, 1]) and not np.array_equal(unique_labels, [0]) and not np.array_equal(unique_labels, [1]):
        raise ValueError(f"Labels must be binary (0,1), got unique values: {unique_labels}")


def validate_config_params(config: Any) -> None:
    """Validate configuration parameters."""
    if hasattr(config, 'model'):
        if config.model.d_model <= 0:
            raise ValueError("d_model must be positive")
        if config.model.nhead <= 0:
            raise ValueError("nhead must be positive")
        if config.model.d_model % config.model.nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
    
    if hasattr(config, 'training'):
        if config.training.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if config.training.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")