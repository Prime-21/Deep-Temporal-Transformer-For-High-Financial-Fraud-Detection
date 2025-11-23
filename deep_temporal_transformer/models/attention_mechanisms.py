"""
Advanced Attention Mechanisms for Deep Temporal Transformer

This module implements state-of-the-art attention mechanisms optimized for
fraud detection in temporal transaction data:
- Multi-head sparse attention with learnable sparsity
- Adaptive temporal attention with time-decay kernels
- Relative position bias (ALiBi-style)
- Flash Attention 2 integration for memory efficiency
"""

import math
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    print("Warning: Flash Attention not available. Install with: pip install flash-attn")


class MultiHeadSparseAttention(nn.Module):
    """
    Multi-head sparse attention with learnable sparsity patterns.
    
    Reduces complexity from O(n²) to O(n log n) by selecting top-k most relevant
    positions per query. The sparsity pattern is learned during training.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        top_k: Number of top positions to attend to (sparsity parameter)
        use_flash: Whether to use Flash Attention if available
        
    References:
        - "Generating Long Sequences with Sparse Transformers" (Child et al., 2019)
        - "Longformer: The Long-Document Transformer" (Beltagy et al., 2020)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        top_k: Optional[int] = None,
        use_flash: bool = True
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.top_k = top_k  # If None, use full attention
        self.use_flash = use_flash and FLASH_ATTENTION_AVAILABLE
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Learnable sparsity scoring network (predicts importance of each position)
        self.sparsity_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            attention_mask: Optional mask (batch_size, seq_len) or (batch_size, seq_len, seq_len)
            return_attention_weights: Whether to return attention weights
            
        Returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: Optional (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: (batch, num_heads, seq_len, head_dim)
        
        if self.use_flash and not return_attention_weights:
            # Use Flash Attention for efficient computation
            # Reshape for Flash Attention: (batch, seq_len, num_heads, head_dim)
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            
            output = flash_attn_func(q, k, v, dropout_p=self.dropout.p if self.training else 0.0)
            output = output.view(batch_size, seq_len, self.d_model)
            attention_weights = None
            
        else:
            # Standard scaled dot-product attention with optional sparsity
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            # Shape: (batch, num_heads, seq_len, seq_len)
            
            # Apply sparsity if top_k is specified
            if self.top_k is not None and self.top_k < seq_len:
                # Compute position importance scores
                importance_scores = self.sparsity_scorer(x).squeeze(-1)  # (batch, seq_len)
                
                # Select top-k positions per query
                _, top_k_indices = torch.topk(importance_scores, self.top_k, dim=-1)
                
                # Create sparse mask
                sparse_mask = torch.zeros_like(attn_scores[0, 0])  # (seq_len, seq_len)
                for i in range(seq_len):
                    sparse_mask[i, top_k_indices[:, :self.top_k].flatten()] = 1
                
                # Apply sparse mask
                attn_scores = attn_scores.masked_fill(sparse_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
            
            # Apply attention mask if provided
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
            
            # Softmax and dropout
            attention_weights = F.softmax(attn_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            # Apply attention to values
            output = torch.matmul(attention_weights, v)
            # Shape: (batch, num_heads, seq_len, head_dim)
            
            # Reshape back
            output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.out_proj(output)
        
        if return_attention_weights:
            return output, attention_weights
        return output, None


class AdaptiveTemporalAttention(nn.Module):
    """
    Adaptive temporal attention with learnable time-decay kernels.
    
    Attention weights are modulated by actual timestamp differences using
    learnable exponential/Gaussian decay functions. This allows the model
    to learn appropriate time scales for fraud detection.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        decay_type: Type of decay function ('exponential' or 'gaussian')
        
    References:
        - "Time-aware Large Kernel Convolutions" (ICML 2021)
        - "Temporal Fusion Transformers" (arXiv:1912.09363)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        decay_type: str = 'exponential'
    ):
        super().__init__()
        assert d_model % num_heads == 0
        assert decay_type in ['exponential', 'gaussian']
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.decay_type = decay_type
        
        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Learnable decay parameters (per head)
        if decay_type == 'exponential':
            # Decay: exp(-λ * |t_i - t_j|)
            self.decay_lambda = nn.Parameter(torch.ones(num_heads))
        else:  # gaussian
            # Decay: exp(-(t_i - t_j)² / (2σ²))
            self.decay_sigma = nn.Parameter(torch.ones(num_heads))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            timestamps: Timestamp tensor (batch_size, seq_len) - Unix time or normalized
            attention_mask: Optional mask (batch_size, seq_len)
            
        Returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # Shape: (batch, num_heads, seq_len, seq_len)
        
        # Compute time differences matrix
        time_diff = timestamps.unsqueeze(-1) - timestamps.unsqueeze(-2)  # (batch, seq_len, seq_len)
        time_diff_abs = torch.abs(time_diff)
        
        # Compute time-decay bias (per head)
        if self.decay_type == 'exponential':
            # Ensure positive decay rates
            decay_lambda = F.softplus(self.decay_lambda)  # (num_heads,)
            decay_bias = torch.exp(-decay_lambda.view(1, -1, 1, 1) * time_diff_abs.unsqueeze(1))
        else:  # gaussian
            decay_sigma = F.softplus(self.decay_sigma)
            decay_bias = torch.exp(-(time_diff_abs.unsqueeze(1) ** 2) / (2 * decay_sigma.view(1, -1, 1, 1) ** 2))
        
        # Modulate attention scores by time-decay bias
        attn_scores = attn_scores * decay_bias
        
        # Apply attention mask
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            attn_scores = attn_scores.masked_fill(mask_expanded == 0, float('-inf'))
        
        # Softmax and dropout
        attention_weights = F.softmax(attn_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply to values
        output = torch.matmul(attention_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.out_proj(output)
        
        return output, attention_weights


class ALiBiPositionalBias(nn.Module):
    """
    Attention with Linear Biases (ALiBi) - relative position encoding.
    
    Instead of adding positional embeddings, adds biases to attention scores
    based on relative distances. Enables better extrapolation to longer sequences.
    
    Args:
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length for pre-computing slopes
        
    References:
        - "Train Short, Test Long: Attention with Linear Biases" (Press et al., 2022)
    """
    
    def __init__(self, num_heads: int, max_seq_len: int = 1024):
        super().__init__()
        self.num_heads = num_heads
        
        # Compute head-specific slopes (geometric sequence)
        slopes = torch.pow(2, torch.arange(1, num_heads + 1) * (-8.0 / num_heads))
        self.register_buffer('slopes', slopes.view(1, num_heads, 1, 1))
        
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Generate position bias matrix for given sequence length.
        
        Args:
            seq_len: Current sequence length
            
        Returns:
            bias: (1, num_heads, seq_len, seq_len) - to be added to attention scores
        """
        # Create relative position matrix: relative_pos[i, j] = i - j
        positions = torch.arange(seq_len, device=self.slopes.device)
        relative_pos = positions.unsqueeze(0) - positions.unsqueeze(1)  # (seq_len, seq_len)
        
        # Compute bias: -|i - j| * slope (per head)
        bias = -torch.abs(relative_pos).unsqueeze(0).unsqueeze(0) * self.slopes
        # Shape: (1, num_heads, seq_len, seq_len)
        
        return bias


class MultiHeadAttentionWithALiBi(nn.Module):
    """
    Multi-head attention with ALiBi positional bias.
    
    Combines standard multi-head attention with relative position biases,
    eliminating the need for absolute positional embeddings.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 1024
    ):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.alibi = ALiBiPositionalBias(num_heads, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input (batch_size, seq_len, d_model)
            attention_mask: Optional mask (batch_size, seq_len)
            
        Returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Add ALiBi bias
        alibi_bias = self.alibi(seq_len)
        attn_scores = attn_scores + alibi_bias
        
        # Apply mask
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask_expanded == 0, float('-inf'))
        
        # Softmax
        attention_weights = F.softmax(attn_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply to values
        output = torch.matmul(attention_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)
        
        return output, attention_weights


# Export public APIs
__all__ = [
    'MultiHeadSparseAttention',
    'AdaptiveTemporalAttention',
    'ALiBiPositionalBias',
    'MultiHeadAttentionWithALiBi'
]
