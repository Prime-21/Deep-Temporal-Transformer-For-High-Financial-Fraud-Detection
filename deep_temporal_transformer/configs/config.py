"""Configuration module for Deep Temporal Transformer."""
import os
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    input_dim: int = 14
    seq_len: int = 8
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 512
    memory_slots: int = 1024
    dropout: float = 0.2
    emb_dims: Tuple[int, ...] = (20000, 5000, 100)


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-4
    patience: int = 10
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    weight_decay: float = 1e-5


@dataclass
class DataConfig:
    """Data configuration parameters."""
    seq_len: int = 8
    train_frac: float = 0.7
    val_frac: float = 0.15
    fraud_ratio: float = 0.001
    n_samples: int = 100000
    random_state: int = 42


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    device: str = "cpu"
    output_dir: str = "outputs"
    random_seed: int = 42
    
    def __post_init__(self):
        """Validate and sanitize configuration."""
        try:
            # Sanitize output directory path
            try:
                from deep_temporal_transformer.utils.security_fixes import validate_path
                self.output_dir = validate_path(self.output_dir)
            except (ImportError, ModuleNotFoundError):
                # Fallback validation
                self.output_dir = os.path.normpath(self.output_dir)
                if os.path.isabs(self.output_dir) or '..' in self.output_dir:
                    raise ValueError("Invalid output directory path")
            
            # Validate model parameters
            if not isinstance(self.model.d_model, int) or self.model.d_model <= 0:
                raise ValueError("d_model must be a positive integer")
            if not isinstance(self.model.nhead, int) or self.model.nhead <= 0:
                raise ValueError("nhead must be a positive integer")
            
            # Ensure model dimensions are compatible
            if self.model.d_model % self.model.nhead != 0:
                raise ValueError("d_model must be divisible by nhead")
                
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {e}")


def get_default_config() -> Config:
    """Get default configuration."""
    try:
        return Config(
            model=ModelConfig(),
            training=TrainingConfig(),
            data=DataConfig()
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create default configuration: {e}")