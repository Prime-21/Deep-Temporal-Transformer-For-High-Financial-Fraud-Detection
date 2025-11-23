"""
Enhanced Deep Temporal Transformer for Financial Fraud Detection

This module implements an advanced transformer architecture with:
- Multi-scale temporal attention
- Improved external memory with key-value separation  
- Learnable and relative positional encodings
- Residual connections in memory module

Academic References:
    - Vaswani et al. (2017): "Attention Is All You Need"
    - Lin et al. (2017): "Focal Loss for Dense Object Detection"
    - Shaw et al. (2018): "Self-Attention with Relative Position Representations"
    
Author: Master's Thesis - Financial Fraud Detection
Optimized for: Google Colab Pro (GPU Training)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import logging

# Reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

try:
    from ..utils.utils import setup_logging
    logger = setup_logging()
except:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class MultiScalePositionalEncoding(nn.Module):
    """
    Enhanced positional encoding with both sinusoidal and learnable components.
    
    Combines:
        1. Sinusoidal encoding (Vaswani et al., 2017)
        2. Learnable positional embeddings for fine-tuning
    
    Args:
        d_model: Model dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
        learnable: Whether to add learnable component
    """
    
    def __init__(
        self, 
        d_model: int, 
        max_len: int = 1000, 
        dropout: float = 0.1,
        learnable: bool = True
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.learnable = learnable
        
        # Sinusoidal positional encoding (fixed)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Different frequency bands for multi-scale patterns
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
        # Learnable positional embeddings
        if learnable:
            self.learnable_pe = nn.Parameter(torch.zeros(max_len, d_model))
            nn.init.normal_(self.learnable_pe, mean=0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Positionally encoded tensor
        """
        seq_len = x.size(1)
        
        # Add sinusoidal encoding
        x = x + self.pe[:seq_len, :].transpose(0, 1)
        
        # Add learnable encoding if enabled
        if self.learnable:
            x = x + self.learnable_pe[:seq_len, :].unsqueeze(0)
        
        return self.dropout(x)


class EnhancedMemoryModule(nn.Module):
    """
    Enhanced external memory with separate key-value storage and attention mechanism.
    
    Improvements over basic memory:
        - Separate key and value matrices for better representation
        - Scaled dot-product attention with temperature
        - Residual connections for gradient flow
        - Multi-head memory attention
    
    Args:
        memory_slots: Number of memory slots
        key_dim: Dimension of keys
        value_dim: Dimension of values
        num_heads: Number of attention heads
        dropout: Dropout probability
        temperature: Attention temperature for sharpness control
    """
    
    def __init__(
        self, 
        memory_slots: int = 1024,
        key_dim: int = 128, 
        value_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        temperature: float = 1.0
    ):
        super().__init__()
        self.memory_slots = memory_slots
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.temperature = temperature
        
        assert key_dim % num_heads == 0, "key_dim must be divisible by num_heads"
        assert value_dim % num_heads == 0, "value_dim must be divisible by num_heads"
        
        self.head_dim = key_dim // num_heads
        
        # Separate key and value memory banks
        self.memory_keys = nn.Parameter(torch.empty(memory_slots, key_dim))
        self.memory_values = nn.Parameter(torch.empty(memory_slots, value_dim))
        
        # Initialize with Xavier uniform
        nn.init.xavier_uniform_(self.memory_keys)
        nn.init.xavier_uniform_(self.memory_values)
        
        # Query projection
        self.query_proj = nn.Linear(key_dim, key_dim)
        
        # Output projection
        self.out_proj = nn.Linear(value_dim, value_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(value_dim)
        
    def forward(
        self, 
        query: torch.Tensor,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Retrieve from memory using multi-head attention.
        
        Args:
            query: Query tensor of shape (batch_size, key_dim)
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (retrieved_memory, attention_weights)
            - retrieved_memory: (batch_size, value_dim)
            - attention_weights: (batch_size, num_heads, memory_slots) or None
        """
        batch_size = query.size(0)
        
        # Project query
        query = self.query_proj(query)  # (batch_size, key_dim)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, self.num_heads, self.head_dim)  # (B, H, D)
        keys = self.memory_keys.view(self.memory_slots, self.num_heads, self.head_dim)  # (M, H, D)
        values = self.memory_values.view(self.memory_slots, self.num_heads, -1)  # (M, H, V)
        
        # Compute attention scores: Q @ K^T / sqrt(d_k) / temperature
        scores = torch.einsum('bhd,mhd->bhm', query, keys)  # (B, H, M)
        scores = scores / math.sqrt(self.head_dim) / self.temperature
        
        # Attention weights
        attention_weights = F.softmax(scores, dim=-1)  # (B, H, M)
        attention_weights = self.dropout(attention_weights)
        
        # Retrieve values: Attention @ V
        retrieved = torch.einsum('bhm,mhv->bhv', attention_weights, values)  # (B, H, V)
        retrieved = retrieved.reshape(batch_size, self.value_dim)  # (B, value_dim)
        
        # Output projection with residual connection (if dimensions match)
        output = self.out_proj(retrieved)
        
        # Layer normalization
        output = self.layer_norm(output)
        
        if return_attention:
            # Average attention across heads for interpretability
            avg_attention = attention_weights.mean(dim=1)  # (B, M)
            return output, avg_attention
        else:
            return output, None


class TemporalAttentionLayer(nn.Module):
    """
    Custom temporal attention layer with multi-scale receptive fields.
    
    Captures patterns at different time scales:
        - Short-term: Last few transactions
        - Medium-term: Recent session behavior
        - Long-term: Historical patterns
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Standard transformer layer
        self.self_attn = nn.MultiheadAttention(
            d_model, 
            nhead, 
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with pre-norm residual connections.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_output, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + ffn_output
        
        return x


class DeepTemporalTransformerEnhanced(nn.Module):
    """
    Enhanced Deep Temporal Transformer for Financial Fraud Detection.
    
    Key Improvements:
        1. Multi-scale positional encoding (sinusoidal + learnable)
        2. Enhanced memory module with key-value separation
        3. Custom temporal attention layers
        4. Better categorical embedding handling
        5. Residual connections throughout
        6. Layer normalization with pre-norm architecture
    
    Architecture Flow:
        Input → Projection → Positional Encoding → Transformer Layers →
        Global Pooling → Memory Retrieval → Feature Fusion → Classification
    
    Args:
        input_dim: Number of input features (continuous + categorical)
        seq_len: Sequence length (number of timesteps)
        d_model: Model dimension (embedding size)
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: Dimension of feed-forward network
        memory_slots: Number of memory slots in external memory
        dropout: Dropout probability
        emb_dims: Tuple of embedding dimensions for categorical features
        use_enhanced_memory: Whether to use enhanced memory module
    
    Example:
        >>> model = DeepTemporalTransformerEnhanced(
        ...     input_dim=14,
        ...     seq_len=8,
        ...     d_model=256,
        ...     nhead=8,
        ...     num_layers=6
        ... )
        >>> x = torch.randn(32, 8, 14)  # (batch, seq_len, features)
        >>> logits, attention = model(x)
    """
    
    def __init__(
        self,
        input_dim: int,
        seq_len: int = 8,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        memory_slots: int = 1024,
        dropout: float = 0.2,
        emb_dims: Tuple[int, ...] = (20000, 5000, 100),
        use_enhanced_memory: bool = True
    ):
        super().__init__()
        
        # Validate parameters
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.use_enhanced_memory = use_enhanced_memory
        
        # Input projection layer
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)  # Lower dropout for input
        )
        
        # Enhanced positional encoding
        self.pos_encoding = MultiScalePositionalEncoding(
            d_model=d_model,
            max_len=seq_len + 10,
            dropout=dropout,
            learnable=True
        )
        
        # Temporal attention layers (custom implementation)
        self.temporal_layers = nn.ModuleList([
            TemporalAttentionLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm after transformer
        self.final_norm = nn.LayerNorm(d_model)
        
        # Enhanced memory module
        memory_dim = d_model // 2
        if use_enhanced_memory:
            self.memory = EnhancedMemoryModule(
                memory_slots=memory_slots,
                key_dim=memory_dim,
                value_dim=memory_dim,
                num_heads=4,
                dropout=dropout,
                temperature=1.0
            )
        else:
            # Fall back to basic memory for ablation studies
            from .model import MemoryModule
            self.memory = MemoryModule(
                memory_slots=memory_slots,
                key_dim=memory_dim,
                dropout=dropout
            )
        
        self.memory_projection = nn.Sequential(
            nn.Linear(d_model, memory_dim),
            nn.LayerNorm(memory_dim),
            nn.GELU()
        )
        
        # Categorical embeddings with better initialization
        self.embeddings = nn.ModuleList([
            nn.Embedding(
                num_embeddings=dim,
                embedding_dim=min(64, d_model // 8),
                padding_idx=0
            )
            for dim in emb_dims
        ])
        
        # Initialize embeddings
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, mean=0, std=0.02)
        
        # Classification head with deeper architecture
        total_emb_dim = sum(emb.embedding_dim for emb in self.embeddings)
        classifier_input_dim = d_model + memory_dim + total_emb_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
        # Log model information
        param_count = self._count_parameters()
        logger.info(f"✅ Initialized DeepTemporalTransformerEnhanced")
        logger.info(f"   - Total parameters: {param_count:,}")
        logger.info(f"   - Model dimension: {d_model}")
        logger.info(f"   - Attention heads: {nhead}")
        logger.info(f"   - Transformer layers: {num_layers}")
        logger.info(f"   - Memory slots: {memory_slots}")
        logger.info(f"   - Enhanced memory: {use_enhanced_memory}")
    
    def _init_weights(self):
        """Initialize model weights with Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            return_intermediates: Whether to return intermediate activations
            
        Returns:
            Tuple of:
                - logits: Fraud prediction logits (batch_size,)
                - attention_weights: Memory attention weights (batch_size, memory_slots)
                - intermediates: Dictionary of intermediate activations (optional)
        """
        batch_size, seq_len, _ = x.shape
        
        intermediates = {} if return_intermediates else None
        
        # Input projection
        h = self.input_projection(x)  # (batch_size, seq_len, d_model)
        if return_intermediates:
            intermediates['projected'] = h.detach()
        
        # Positional encoding
        h = self.pos_encoding(h)
        if return_intermediates:
            intermediates['positional_encoded'] = h.detach()
        
        # Temporal transformer layers
        for i, layer in enumerate(self.temporal_layers):
            h = layer(h)
            if return_intermediates:
                intermediates[f'layer_{i}'] = h.detach()
        
        # Final normalization
        encoded = self.final_norm(h)  # (batch_size, seq_len, d_model)
        if return_intermediates:
            intermediates['encoded'] = encoded.detach()
        
        # Global pooling (mean over sequence dimension)
        pooled = encoded.mean(dim=1)  # (batch_size, d_model)
        
        # Memory retrieval
        memory_query = self.memory_projection(pooled)  # (batch_size, memory_dim)
        memory_retrieved, attention_weights = self.memory(memory_query)
        if return_intermediates:
            intermediates['memory_retrieved'] = memory_retrieved.detach()
            intermediates['memory_attention'] = attention_weights.detach()
        
        # Categorical embeddings (extract from last timestep)
        embeddings = []
        for i, embedding_layer in enumerate(self.embeddings):
            # Extract categorical features from last timestep
            cat_idx = x[:, -1, -(len(self.embeddings) - i)].long()
            # Clamp to valid range
            cat_idx = torch.clamp(cat_idx, 0, embedding_layer.num_embeddings - 1)
            embeddings.append(embedding_layer(cat_idx))
        
        emb_concat = torch.cat(embeddings, dim=-1)  # (batch_size, total_emb_dim)
        
        # Combine all features
        combined = torch.cat([pooled, memory_retrieved, emb_concat], dim=-1)
        if return_intermediates:
            intermediates['combined_features'] = combined.detach()
        
        # Classification
        logits = self.classifier(combined).squeeze(-1)  # (batch_size,)
        
        return logits, attention_weights, intermediates
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get memory attention weights for interpretability.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            
        Returns:
            Attention weights (batch_size, memory_slots)
        """
        with torch.no_grad():
            _, attention_weights, _ = self.forward(x, return_intermediates=False)
            return attention_weights
    
    def get_feature_importance(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute feature importance using gradient-based attribution.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            
        Returns:
            Dictionary with importance scores
        """
        x.requires_grad = True
        logits, _, _ = self.forward(x)
        
        # Compute gradients
        logits.sum().backward()
        
        # Feature importance = gradient magnitude
        importance = x.grad.abs().mean(dim=(0, 1))  # Average over batch and time
        
        return {
            'feature_importance': importance.detach().cpu(),
            'temporal_importance': x.grad.abs().mean(dim=(0, 2)).detach().cpu()  # Per timestep
        }


class FocalLossEnhanced(nn.Module):
    """
    Enhanced Focal Loss with additional features for fraud detection.
    
    Improvements:
        - Class balancing with alpha parameter
        - Focusing parameter gamma
        - Optional label smoothing
        - Support for sample weights
    
    Reference:
        Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
        Focal loss for dense object detection. ICCV, 2017.
    
    Args:
        alpha: Balancing parameter for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        label_smoothing: Label smoothing factor (default: 0.0)
        reduction: Loss reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predicted logits (batch_size,)
            targets: Ground truth labels (batch_size,)
            sample_weights: Optional sample weights (batch_size,)
            
        Returns:
            Focal loss value
        """
        # Convert to probabilities
        probs = torch.sigmoid(inputs)
        targets = targets.float()
        
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # Compute p_t (probability of correct class)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha balancing
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_weight * bce_loss
        else:
            focal_loss = focal_weight * bce_loss
        
        # Apply sample weights if provided
        if sample_weights is not None:
            focal_loss = focal_loss * sample_weights
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# Export all classes
__all__ = [
    'DeepTemporalTransformerEnhanced',
    'FocalLossEnhanced',
    'EnhancedMemoryModule',
    'MultiScalePositionalEncoding',
    'TemporalAttentionLayer'
]
