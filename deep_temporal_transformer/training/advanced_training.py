"""
Advanced Training Utilities for Colab Pro GPU Optimization

This module provides production-ready training utilities optimized for Google Colab Pro:
- GPU hardware detection and optimization
- Mixed precision training (FP16/BF16)
- Advanced loss functions (Focal, Class-Balanced, Contrastive)
- Curriculum learning
- Learning rate scheduling
- Gradient checkpointing management
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, Tuple, List
import math


def detect_and_configure_gpu() -> Dict[str, any]:
    """
    Detect GPU hardware and return optimal configuration.
    
    Returns:
        config: Dictionary with device, dtype, and optimization settings
    """
    config = {
        'device': torch.device('cpu'),
        'gpu_name': None,
        'gpu_memory_gb': 0,
        'use_fp16': False,
        'use_bf16': False,
        'use_tf32': False,
        'batch_size_multiplier': 1.0
    }
    
    if torch.cuda.is_available():
        config['device'] = torch.device('cuda')
        config['gpu_name'] = torch.cuda.get_device_name(0)
        config['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # GPU-specific optimizations
        if 'A100' in config['gpu_name']:
            config['use_bf16'] = True  # BF16 is optimal for A100
            config['use_tf32'] = True   # Enable TF32 for faster matmul
            config['batch_size_multiplier'] = 2.0
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        elif 'V100' in config['gpu_name']:
            config['use_fp16'] = True
            config['batch_size_multiplier'] = 1.5
        elif 'T4' in config['gpu_name']:
            config['use_fp16'] = True
            config['batch_size_multiplier'] = 1.0
        else:
            # Default for unknown GPUs
            config['use_fp16'] = True
            config['batch_size_multiplier'] = 1.0
        
        print(f"🚀 GPU Detected: {config['gpu_name']}")
        print(f"💾 GPU Memory: {config['gpu_memory_gb']:.1f} GB")
        print(f"⚡ Mixed Precision: {'BF16' if config['use_bf16'] else ('FP16' if config['use_fp16'] else 'FP32')}")
        print(f"📊 Recommended Batch Size Multiplier: {config['batch_size_multiplier']:.1f}x")
    else:
        print("⚠️  No GPU detected, using CPU")
    
    return config


class FocalLossAdvanced(nn.Module):
    """
    Advanced Focal Loss with auto-tuned gamma and class weights.
    
    Focal Loss = -α(1-p_t)^γ log(p_t)
    
    Args:
        alpha: Weighting factor for class imbalance (0-1)
        gamma: Focusing parameter (higher = more focus on hard examples)
        auto_tune_gamma: Whether to learn gamma during training
        label_smoothing: Label smoothing factor (0-1)
        
    References:
        - "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        auto_tune_gamma: bool = False,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        
        if auto_tune_gamma:
            self.gamma = nn.Parameter(torch.tensor(gamma))
        else:
            self.register_buffer('gamma', torch.tensor(gamma))
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size,) - model logits
            targets: (batch_size,) - binary labels {0, 1}
            
        Returns:
            loss: Scalar focal loss
        """
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Compute probabilities
        probs = torch.sigmoid(logits)
        targets = targets.float()
        
        # Compute p_t (probability of correct class)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** torch.abs(self.gamma)  # abs for learnable gamma
        
        # Compute alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        #Focal loss
        loss = alpha_t * focal_weight * bce
        
        return loss.mean()


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss considering effective number of samples.
    
    Re-weights samples based on effective class sizes to handle extreme imbalance.
    
    Args:
        samples_per_class: List/tensor of sample counts per class
        beta: Hyperparameter (0-1), typically 0.9999 for extreme imbalance
        loss_type: Base loss type ('focal' or 'ce')
        
    References:
        - "Class-Balanced Loss Based on Effective Number of Samples" (Cui et al., 2019)
    """
    
    def __init__(
        self,
        samples_per_class: List[int],
        beta: float = 0.9999,
        loss_type: str = 'focal'
    ):
        super().__init__()
        assert loss_type in ['focal', 'ce']
        
        # Compute effective number of samples
        effective_num = 1.0 - torch.pow(beta, torch.tensor(samples_per_class, dtype=torch.float32))
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(weights)  # Normalize
        
        self.register_buffer('weights', weights)
        self.loss_type = loss_type
        
        if loss_type == 'focal':
            self.base_loss = FocalLossAdvanced()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size,)
            targets: (batch_size,)
            
        Returns:
            loss: Scalar
        """
        if self.loss_type == 'focal':
            loss = self.base_loss(logits, targets)
        else:
            targets_float = targets.float()
            loss = F.binary_cross_entropy_with_logits(logits, targets_float, reduction='none')
        
        # Apply class weights
        sample_weights = self.weights[targets.long()]
        weighted_loss = (loss * sample_weights).mean()
        
        return weighted_loss


class ContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss for learning discriminative embeddings.
    
    Pulls together embeddings of same class, pushes apart different classes.
    
    Args:
        temperature: Temperature parameter for scaling
        
    References:
        - "Supervised Contrastive Learning" (Khosla et al., 2020)
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch_size, embedding_dim) - L2 normalized embeddings
            labels: (batch_size,) - binary labels
            
        Returns:
            loss: Scalar contrastive loss
        """
        batch_size = embeddings.shape[0]
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create mask for positive pairs (same class)
        labels = labels.view(-1, 1)
        mask_positive = (labels == labels.T).float()
        mask_positive.fill_diagonal_(0)  # Exclude self-comparisons
        
        # Compute log probabilities
        exp_sim = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # Mean of log-likelihood over positive pairs
        mean_log_prob = (mask_positive * log_prob).sum(dim=1) / mask_positive.sum(dim=1).clamp(min=1)
        
        loss = -mean_log_prob.mean()
        
        return loss


class CurriculumScheduler:
    """
    Curriculum learning scheduler for progressive difficulty training.
    
    Gradually increases task difficulty over training:
    - Easy examples (clear fraud/legitimate) → Hard examples (edge cases)
    - Short sequences → Long sequences
    - Low fraud rate → Realistic fraud rate
    
    Args:
        total_epochs: Total training epochs
        start_difficulty: Initial difficulty (0-1)
        end_difficulty: Final difficulty (0-1)
        warmup_epochs: Number of warmup epochs
    """
    
    def __init__(
        self,
        total_epochs: int,
        start_difficulty: float = 0.3,
        end_difficulty: float = 1.0,
        warmup_epochs: int = 5
    ):
        self.total_epochs = total_epochs
        self.start_difficulty = start_difficulty
        self.end_difficulty = end_difficulty
        self.warmup_epochs = warmup_epochs
    
    def get_difficulty(self, epoch: int) -> float:
        """Get current difficulty level."""
        if epoch < self.warmup_epochs:
            # Linear warmup
            progress = epoch / self.warmup_epochs
        else:
            # Cosine annealing to end difficulty
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            progress = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
            progress = 1 - progress  # Invert for increasing difficulty
        
        difficulty = self.start_difficulty + progress * (self.end_difficulty - self.start_difficulty)
        return min(max(difficulty, 0.0), 1.0)
    
    def filter_by_difficulty(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        difficulty: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Filter dataset by difficulty based on model uncertainty.
        
        Args:
            data: Input data
            labels: Ground truth labels
            predictions: Model predictions (probabilities)
            difficulty: Current difficulty (0-1)
            
        Returns:
            filtered_data, filtered_labels
        """
        # Compute sample difficulty (higher = harder)
        # Hard samples are those where prediction is uncertain
        sample_difficulty = 1 - torch.abs(predictions - labels.float())
        
        # Select samples below difficulty threshold
        threshold = torch.quantile(sample_difficulty, difficulty)
        mask = sample_difficulty <= threshold
        
        return data[mask], labels[mask]


class WarmupCosineScheduler:
    """
    Learning rate scheduler with warmup and cosine annealing with restarts.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs
        max_epochs: Total epochs
        base_lr: Base learning rate
        min_lr: Minimum learning rate
        restart_every: Epochs between restarts (0 = no restarts)
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        base_lr: float,
        min_lr: float = 1e-6,
        restart_every: int = 0
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.restart_every = restart_every
        self.current_epoch = 0
    
    def step(self, epoch: Optional[int] = None):
        """Update learning rate."""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing
            progress = self.current_epoch - self.warmup_epochs
            total_progress = self.max_epochs - self.warmup_epochs
            
            if self.restart_every > 0:
                # With restarts
                progress = progress % self.restart_every
                total_progress = self.restart_every
            
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress / total_progress))
            lr = self.min_lr + (self.base_lr - self.min_lr) * cosine_decay
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


def optimize_memory():
    """Clear GPU memory and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    import gc
    gc.collect()


def print_memory_stats():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        
        print(f"💾 GPU Memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Peak={max_allocated:.2f}GB")


# Export
__all__ = [
    'detect_and_configure_gpu',
    'FocalLossAdvanced',
    'ClassBalancedLoss',
    'ContrastiveLoss',
    'CurriculumScheduler',
    'WarmupCosineScheduler',
    'optimize_memory',
    'print_memory_stats'
]
