"""
Temporal Processing Modules for Deep Temporal Transformer

This module implements specialized temporal components:
- Hierarchical Temporal Pyramid Network
- Temporal Convolutional Network (TCN) encoder
- Multi-scale temporal aggregation
- Temporal consistency regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict


class TemporalConvBlock(nn.Module):
    """
    Temporal convolutional block with dilated causal convolutions.
    
    Uses exponentially increasing dilation rates to capture long-range
    dependencies efficiently without attention.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size
        dilation: Dilation rate
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Calculate padding for causal convolution
        self.padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation
        )
        
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, in_channels, seq_len)
            
        Returns:
            (batch_size, out_channels, seq_len)
        """
        residual = self.residual(x)
        
        # First conv block
        out = self.conv1(x)
        # Remove future padding (causal)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = out.transpose(1, 2)  # (batch, seq_len, channels) for LayerNorm
        out = self.norm1(out)
        out = out.transpose(1, 2)  # Back to (batch, channels, seq_len)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Second conv block
        out = self.conv2(out)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = out.transpose(1, 2)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Residual connection
        return out + residual


class TemporalConvolutionalNetwork(nn.Module):
    """
    Temporal Convolutional Network (TCN) for sequence encoding.
    
    Stacks multiple dilated causal convolution layers with exponentially
    increasing receptive field. Captures long-range dependencies efficiently.
    
    Args:
        input_dim: Input feature dimension
        hidden_channels: List of channel sizes for each TCN block
        kernel_size: Convolution kernel size
        dropout: Dropout probability
        
    References:
        - "An Empirical Evaluation of Generic Convolutional and Recurrent Networks"
          (Bai et al., 2018)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_channels: List[int] = [128, 256, 256, 512],
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = hidden_channels[-1]
        
        layers = []
        num_layers = len(hidden_channels)
        
        for i in range(num_layers):
            in_ch = input_dim if i == 0 else hidden_channels[i - 1]
            out_ch = hidden_channels[i]
            dilation = 2 ** i  # Exponentially increasing dilation
            
            layers.append(
                TemporalConvBlock(
                    in_ch,
                    out_ch,
                    kernel_size,
                    dilation,
                    dropout
                )
            )
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            
        Returns:
            (batch_size, seq_len, output_dim)
        """
        # Transpose for Conv1d: (batch, channels, seq_len)
        x = x.transpose(1, 2)
        out = self.network(x)
        # Transpose back: (batch, seq_len, channels)
        out = out.transpose(1, 2)
        return out


class HierarchicalTemporalPyramid(nn.Module):
    """
    Hierarchical Temporal Pyramid Network for multi-resolution processing.
    
    Processes input at multiple temporal resolutions (e.g., 1min, 5min, 15min, 1hr)
    and fuses information across scales. This captures both fine-grained and
    coarse-grained temporal patterns.
    
    Args:
        d_model: Model dimension
        window_sizes: List of aggregation window sizes (in timesteps)
        pooling_type: Type of pooling ('avg', 'max', or 'attention')
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        d_model: int,
        window_sizes: List[int] = [1, 5, 15, 30],  # Multi-resolution windows
        pooling_type: str = 'attention',
        dropout: float = 0.1
    ):
        super().__init__()
        assert pooling_type in ['avg', 'max', 'attention']
        
        self.d_model = d_model
        self.window_sizes = sorted(window_sizes)
        self.pooling_type = pooling_type
        self.num_scales = len(window_sizes)
        
        # Per-scale processing
        self.scale_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            for _ in window_sizes
        ])
        
        # Attention-based pooling (if selected)
        if pooling_type == 'attention':
            self.pool_attention = nn.ModuleList([
                nn.Linear(d_model, 1) for _ in window_sizes
            ])
        
        # Cross-scale fusion
        fusion_input_dim = d_model * self.num_scales
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
    def _pool_window(
        self,
        x: torch.Tensor,
        window_size: int,
        scale_idx: int
    ) -> torch.Tensor:
        """
        Pool features over specified window size.
        
        Args:
            x: (batch_size, seq_len, d_model)
            window_size: Size of pooling window
            scale_idx: Index of the scale (for attention network selection)
            
        Returns:
            (batch_size, seq_len, d_model) - pooled features
        """
        batch_size, seq_len, d_model = x.shape
        
        if window_size == 1:
            return x  # No pooling for finest resolution
        
        # Unfold to create windows: (batch, num_windows, window_size, d_model)
        # Pad if necessary
        pad_len = (window_size - seq_len % window_size) % window_size
        if pad_len > 0:
            x_padded = F.pad(x, (0, 0, 0, pad_len))
        else:
            x_padded = x
        
        new_seq_len = x_padded.shape[1]
        x_windowed = x_padded.view(batch_size, new_seq_len // window_size, window_size, d_model)
        
        # Apply pooling
        if self.pooling_type == 'avg':
            pooled = x_windowed.mean(dim=2)  # (batch, num_windows, d_model)
        elif self.pooling_type == 'max':
            pooled = x_windowed.max(dim=2)[0]
        else:  # attention
            # Compute attention weights over window
            attn_scores = self.pool_attention[scale_idx](x_windowed).squeeze(-1)  # (batch, num_windows, window_size)
            attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)  # (batch, num_windows, window_size, 1)
            pooled = (x_windowed * attn_weights).sum(dim=2)  # (batch, num_windows, d_model)
        
        # Repeat to match original sequence length
        pooled_repeated = pooled.repeat_interleave(window_size, dim=1)[:, :seq_len, :]
        
        return pooled_repeated
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            
        Returns:
            (batch_size, seq_len, d_model) - multi-scale fused features
        """
        scale_features = []
        
        # Process at each temporal scale
        for scale_idx, window_size in enumerate(self.window_sizes):
            # Pool over window
            pooled = self._pool_window(x, window_size, scale_idx)
            
            # Scale-specific processing
            processed = self.scale_processors[scale_idx](pooled)
            scale_features.append(processed)
        
        # Concatenate all scales
        multi_scale = torch.cat(scale_features, dim=-1)  # (batch, seq_len, d_model * num_scales)
        
        # Cross-scale fusion
        fused = self.fusion_network(multi_scale)  # (batch, seq_len, d_model)
        
        # Residual connection with original input
        output = fused + x
        
        return output


class TemporalConsistencyRegularizer(nn.Module):
    """
    Temporal consistency regularization for stable predictions.
    
    Encourages the model to make smooth predictions across sliding windows,
    penalizing large prediction changes for similar input sequences.
    
    This is used as an auxiliary loss during training.
    """
    
    def __init__(self, lambda_consistency: float = 0.1, distance_type: str = 'l2'):
        super().__init__()
        assert distance_type in ['l2', 'l1', 'cosine']
        
        self.lambda_consistency = lambda_consistency
        self.distance_type = distance_type
        
    def forward(
        self,
        predictions: torch.Tensor,
        window_stride: int = 1
    ) -> torch.Tensor:
        """
        Compute temporal consistency loss.
        
        Args:
            predictions: (batch_size, seq_len) - model predictions over time
            window_stride: Stride between windows to compare
            
        Returns:
            Scalar consistency loss
        """
        # Compare predictions at t and t+stride
        pred_t = predictions[:, :-window_stride]
        pred_t_stride = predictions[:, window_stride:]
        
        if self.distance_type == 'l2':
            consistency_loss = F.mse_loss(pred_t, pred_t_stride)
        elif self.distance_type == 'l1':
            consistency_loss = F.l1_loss(pred_t, pred_t_stride)
        else:  # cosine
            consistency_loss = 1 - F.cosine_similarity(pred_t, pred_t_stride, dim=-1).mean()
        
        return self.lambda_consistency * consistency_loss


class MultiScaleTemporalEncoder(nn.Module):
    """
    Combined multi-scale temporal encoder.
    
    Integrates TCN for local patterns and hierarchical pyramid for
    multi-resolution processing.
    
    Args:
        input_dim: Input feature dimension
        d_model: Model dimension
        tcn_channels: Channel sizes for TCN blocks
        pyramid_windows: Window sizes for temporal pyramid
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        tcn_channels: List[int] = [128, 256, 256],
        pyramid_windows: List[int] = [1, 5, 15, 30],
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # TCN for local temporal patterns
        self.tcn = TemporalConvolutionalNetwork(
            d_model,
            tcn_channels,
            kernel_size=3,
            dropout=dropout
        )
        
        # Hierarchical pyramid for multi-scale patterns
        self.pyramid = HierarchicalTemporalPyramid(
            tcn_channels[-1],
            pyramid_windows,
            pooling_type='attention',
            dropout=dropout
        )
        
        # Final projection to d_model
        self.output_projection = nn.Linear(tcn_channels[-1], d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            
        Returns:
            (batch_size, seq_len, d_model)
        """
        # Initial projection
        x = self.input_projection(x)
        
        # TCN encoding
        x = self.tcn(x)
        
        # Multi-scale pyramid
        x = self.pyramid(x)
        
        # Final projection
        x = self.output_projection(x)
        
        return x


# Export public APIs
__all__ = [
    'TemporalConvBlock',
    'TemporalConvolutionalNetwork',
    'HierarchicalTemporalPyramid',
    'TemporalConsistencyRegularizer',
    'MultiScaleTemporalEncoder'
]
