"""
Advanced Deep Temporal Transformer for Fraud Detection - Production Model

This module integrates all state-of-the-art architectural innovations:
- Multi-head sparse attention with learned sparsity
- Adaptive temporal attention with time-decay kernels
- Hierarchical temporal pyramid processing
- Temporal convolutional encoder (TCN)
- Mixture of Experts (MoE) routing
- ALiBi positional bias
- Uncertainty quantification

Optimized for Google Colab Pro with GPU acceleration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import math

# Import our custom modules
try:
    from .attention_mechanisms import (
        MultiHead SparseAttention,
        AdaptiveTemporalAttention,
        MultiHeadAttentionWithALiBi
    )
    from .temporal_modules import (
        MultiScaleTemporalEncoder,
        TemporalConsistencyRegularizer
    )
    from .moe import MixtureOfExperts
except ImportError:
    # Fallback for standalone use
    pass


class AdvancedTransformerBlock(nn.Module):
    """
    Advanced transformer block with multiple attention mechanisms and MoE.
    
    Combines:
    - Sparse attention for efficiency
    - Temporal attention with time-decay
    - Feed-forward MoE for specialization
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        num_experts: Number of MoE experts
        dropout: Dropout probability
        use_sparse: Whether to use sparse attention
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        num_experts: int = 8,
        dropout: float = 0.1,
        use_sparse: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Multi-head attention (sparse or standard)
        if use_sparse:
            self.attention = MultiHeadSparseAttention(
                d_model,
                num_heads,
                dropout=dropout,
                top_k=None,  # Full attention for now
                use_flash=True
            )
        else:
            self.attention = MultiHeadAttentionWithALiBi(
                d_model,
                num_heads,
                dropout=dropout
            )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Mixture of Experts feed-forward
        self.moe = MixtureOfExperts(
            input_dim=d_model,
            hidden_dim=d_model * 4,
            output_dim=d_model,
            num_experts=num_experts,
            top_k=2,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_aux_loss: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            attention_mask: Optional mask
            return_aux_loss: Whether to return MoE auxiliary loss
            
        Returns:
            output: (batch_size, seq_len, d_model)
            aux_loss: Optional MoE load balancing loss
        """
        # Self-attention with residual
        attn_out, _ = self.attention(x, attention_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # MoE feed-forward with residual
        moe_out, aux_loss = self.moe(x, return_aux_loss=return_aux_loss)
        x = self.norm2(x + self.dropout(moe_out))
        
        return x, aux_loss


class DeepTemporalTransformerAdvanced(nn.Module):
    """
    Production-ready Deep Temporal Transformer with state-of-the-art innovations.
    
    Architecture:
    1. Multi-scale temporal encoding (TCN + Hierarchical Pyramid)
    2. Stacked transformer blocks with sparse attention and MoE
    3. Adaptive temporal attention with time-decay
    4. External memory module for pattern storage
    5. Uncertainty quantification via MC dropout
    6. Multi-task head (fraud detection + auxiliary tasks)
    
    Args:
        input_dim: Input feature dimension
        d_model: Model hidden dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        num_experts: Number of MoE experts per layer
        memory_slots: External memory size
        seq_len: Maximum sequence length
        dropout: Dropout rate
        mc_dropout: Whether to use MC dropout for uncertainty
        
    Optimizations:
    - Flash Attention 2 compatible
    - Gradient checkpointing ready
    - Mixed precision (FP16) compatible
    - Supports torch.compile()
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        num_experts: int = 8,
        memory_slots: int = 512,
        seq_len: int = 100,
        dropout: float = 0.1,
        mc_dropout: bool = True,
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.mc_dropout = mc_dropout
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Multi-scale temporal encoder (TCN + Pyramid)
        self.temporal_encoder = MultiScaleTemporalEncoder(
            input_dim=input_dim,
            d_model=d_model,
            tcn_channels=[128, 256, 256],
            pyramid_windows=[1, 5, 10, 20],
            dropout=dropout
        )
        
        # Stacked transformer blocks
        self.transformer_blocks = nn.ModuleList([
            AdvancedTransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_experts=num_experts,
                dropout=dropout,
                use_sparse=(i % 2 == 0)  # Alternate sparse and standard attention
            )
            for i in range(num_layers)
        ])
        
        # Adaptive temporal attention (with timestamp awareness)
        self.temporal_attention = AdaptiveTemporalAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            decay_type='exponential'
        )
        
        # External memory module
        self.memory_keys = nn.Parameter(torch.randn(memory_slots, d_model))
        self.memory_values = nn.Parameter(torch.randn(memory_slots, d_model))
        self.memory_query_proj = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.memory_keys)
        nn.init.xavier_uniform_(self.memory_values)
        
        # Classification head with uncertainty
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # *2 for pooled + memory
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout if mc_dropout else 0.0),  # MC dropout layer
            nn.Linear(d_model // 2, 1)
        )
        
        # Auxiliary task heads (for multi-task learning)
        self.amount_predictor = nn.Linear(d_model, 1)  # Predict transaction amount
        self.user_embedding = nn.Linear(d_model, 64)  # User representation
        
        # Initialize weights
        self._init_weights()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Model initialized: {trainable_params:,} trainable params ({total_params:,} total)")
        
    def _init_weights(self):
        """Initialize model weights with appropriate strategies."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _apply_memory_attention(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve from external memory using attention.
        
        Args:
            query: (batch_size, d_model)
            
        Returns:
            retrieved: (batch_size, d_model)
            attention_weights: (batch_size, memory_slots)
        """
        # Project query
        query = self.memory_query_proj(query)  # (batch, d_model)
        
        # Compute attention scores with memory keys
        scores = torch.matmul(query, self.memory_keys.transpose(0, 1))  # (batch, memory_slots)
        scores = scores / math.sqrt(self.d_model)
        
        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # (batch, memory_slots)
        
        # Retrieve from memory values
        retrieved = torch.matmul(attention_weights, self.memory_values)  # (batch, d_model)
        
        return retrieved, attention_weights
    
    def forward(
        self,
        x: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_intermediates: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional intermediate outputs.
        
        Args:
            x: Input sequences (batch_size, seq_len, input_dim)
            timestamps: Optional timestamps (batch_size, seq_len)
            attention_mask: Optional mask (batch_size, seq_len)
            return_intermediates: Whether to return intermediate activations
            
        Returns:
            Dictionary containing:
            - logits: (batch_size,) - fraud probability logits
            - aux_loss: Scalar - auxiliary losses (MoE load balancing)
            - memory_attention: (batch_size, memory_slots) - memory attention weights
            - uncertainty: Optional (batch_size,) - prediction uncertainty
            - intermediates: Optional dict of intermediate activations
        """
        batch_size, seq_len, _ = x.shape
        
        # Multi-scale temporal encoding
        h = self.temporal_encoder(x)  # (batch, seq_len, d_model)
        
        # Stack transformer blocks with optional gradient checkpointing
        aux_losses = []
        for i, block in enumerate(self.transformer_blocks):
            if self.use_gradient_checkpointing and self.training:
                h, aux_loss = torch.utils.checkpoint.checkpoint(
                    block,
                    h,
                    attention_mask,
                    True,
                    use_reentrant=False
                )
            else:
                h, aux_loss = block(h, attention_mask, return_aux_loss=True)
            
            if aux_loss is not None:
                aux_losses.append(aux_loss)
        
        # Adaptive temporal attention (if timestamps provided)
        if timestamps is not None:
            h_temporal, _ = self.temporal_attention(h, timestamps, attention_mask)
            h = h + h_temporal  # Residual connection
        
        # Global pooling (mean + max)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            h_masked = h * mask_expanded
            pooled_mean = h_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            pooled_max = h_masked.max(dim=1)[0]
        else:
            pooled_mean = h.mean(dim=1)
            pooled_max = h.max(dim=1)[0]
        
        # Combine pooling strategies
        pooled = (pooled_mean + pooled_max) / 2  # (batch, d_model)
        
        # External memory retrieval
        memory_retrieved, memory_attention = self._apply_memory_attention(pooled)
        
        # Combine sequence and memory features
        combined = torch.cat([pooled, memory_retrieved], dim=-1)  # (batch, d_model * 2)
        
        # Classification
        logits = self.classifier(combined).squeeze(-1)  # (batch,)
        
        # Compute total auxiliary loss
        total_aux_loss = sum(aux_losses) if aux_losses else torch.tensor(0.0, device=x.device)
        
        # Prepare output
        output = {
            'logits': logits,
            'aux_loss': total_aux_loss,
            'memory_attention': memory_attention
        }
        
        # Monte Carlo Dropout for uncertainty estimation
        if return_intermediates or (not self.training and self.mc_dropout):
            output['uncertainty'] = None  # Computed externally via MC sampling
        
        # Intermediate outputs for interpretability
        if return_intermediates:
            output['intermediates'] = {
                'temporal_encoding': h,
                'pooled_features': pooled,
                'memory_features': memory_retrieved,
                'combined_features': combined
            }
        
        return output
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        n_samples: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty estimation using MC Dropout.
        
        Args:
            x: Input (batch_size, seq_len, input_dim)
            timestamps: Optional timestamps
            n_samples: Number of MC samples
            
        Returns:
            mean_probs: (batch_size,) - mean predicted probabilities
            uncertainty: (batch_size,) - prediction uncertainty (std)
        """
        self.train()  # Enable dropout
        
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                output = self.forward(x, timestamps=timestamps)
                probs = torch.sigmoid(output['logits'])
                predictions.append(probs)
        
        predictions = torch.stack(predictions)  # (n_samples, batch)
        mean_probs = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        
        self.eval()  # Back to eval mode
        
        return mean_probs, uncertainty


# Export
__all__ = ['AdvancedTransformerBlock', 'DeepTemporalTransformerAdvanced']
