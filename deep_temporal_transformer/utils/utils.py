"""Utility functions for Deep Temporal Transformer."""
import os
import json
import random
import logging
import numpy as np
import torch
from typing import Any, Dict, Optional
from pathlib import Path


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    try:
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if level.upper() not in valid_levels:
            level = 'INFO'
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    except Exception:
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> None:
    """Safely create directory with path validation."""
    try:
        # Normalize and validate path
        from .security_fixes import validate_path
        normalized_path = validate_path(path)
        
        # Create directory
        Path(normalized_path).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise OSError(f"Failed to create directory {path}: {e}")


def save_json(obj: Dict[str, Any], path: str) -> None:
    """Safely save JSON with path validation."""
    try:
        # Validate path
        from .security_fixes import validate_path
        normalized_path = validate_path(path, ['.json'])
        
        # Ensure directory exists
        ensure_dir(os.path.dirname(normalized_path))
        
        # Save JSON
        with open(normalized_path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise IOError(f"Failed to save JSON to {path}: {e}")


def load_json(path: str) -> Dict[str, Any]:
    """Safely load JSON with path validation."""
    try:
        # Validate path
        from .security_fixes import validate_path
        normalized_path = validate_path(path, ['.json'])
        if not os.path.exists(normalized_path):
            raise FileNotFoundError(f"File not found: {normalized_path}")
        
        # Load JSON
        with open(normalized_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise IOError(f"Failed to load JSON from {path}: {e}")


def get_device() -> torch.device:
    """Get optimal device for computation."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EarlyStopping:
    """Early stopping utility class."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """Check if training should stop."""
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop