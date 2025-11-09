"""Deep Temporal Transformer model for financial fraud detection."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

# Set PyTorch seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

from .utils import setup_logging

logger = setup_logging()


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].transpose(0, 1)
        return self.dropout(x)


class MemoryModule(nn.Module):
    """External memory module for pattern storage and retrieval."""
    
    def __init__(self, memory_slots: int = 1024, key_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.memory_slots = memory_slots
        self.key_dim = key_dim
        
        # Initialize memory with Xavier uniform
        self.memory = nn.Parameter(torch.empty(memory_slots, key_dim))
        nn.init.xavier_uniform_(self.memory)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve from memory using attention mechanism.
        
        Args:
            query: Query tensor of shape (batch_size, key_dim)
            
        Returns:
            Tuple of (retrieved_memory, attention_weights)
        """
        batch_size = query.size(0)
        
        # Compute attention scores
        scores = torch.matmul(query, self.memory.transpose(0, 1))  # (batch_size, memory_slots)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Retrieve weighted memory
        retrieved = torch.matmul(attention_weights, self.memory)  # (batch_size, key_dim)
        
        return retrieved, attention_weights


class DeepTemporalTransformer(nn.Module):
    """
    Deep Temporal Transformer for high-frequency financial fraud detection.
    
    Architecture:
    1. Input projection and positional encoding
    2. Multi-layer transformer encoder
    3. External memory module for pattern storage
    4. Categorical embeddings for user/device/merchant
    5. Final classification head with focal loss support
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
        emb_dims: Tuple[int, ...] = (20000, 5000, 100)
    ):
        super().__init__()
        
        # Validate parameters
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len + 10, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Memory module
        memory_dim = d_model // 2
        self.memory = MemoryModule(
            memory_slots=memory_slots, 
            key_dim=memory_dim, 
            dropout=dropout
        )
        self.memory_projection = nn.Linear(d_model, memory_dim)
        
        # Categorical embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=dim, embedding_dim=min(64, d_model // 8))
            for dim in emb_dims
        ])
        
        # Classification head
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
            nn.Linear(256, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized DeepTemporalTransformer with {self._count_parameters():,} parameters")
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
    
    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Tuple of (logits, attention_weights)
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection and positional encoding
        h = self.input_projection(x)  # (batch_size, seq_len, d_model)
        h = self.pos_encoding(h)
        
        # Transformer encoding
        encoded = self.transformer(h)  # (batch_size, seq_len, d_model)
        
        # Global pooling (mean over sequence)
        pooled = encoded.mean(dim=1)  # (batch_size, d_model)
        
        # Memory retrieval
        memory_query = self.memory_projection(pooled)  # (batch_size, memory_dim)
        memory_retrieved, attention_weights = self.memory(memory_query)
        
        # Categorical embeddings (from last timestep)
        embeddings = []
        for i, embedding_layer in enumerate(self.embeddings):
            # Extract categorical features from last timestep
            cat_idx = x[:, -1, -(len(self.embeddings) - i)].long()
            cat_idx = torch.clamp(cat_idx, 0, embedding_layer.num_embeddings - 1)
            embeddings.append(embedding_layer(cat_idx))
        
        emb_concat = torch.cat(embeddings, dim=-1)  # (batch_size, total_emb_dim)
        
        # Combine all features
        combined = torch.cat([pooled, memory_retrieved, emb_concat], dim=-1)
        
        # Classification
        logits = self.classifier(combined).squeeze(-1)  # (batch_size,)
        
        return logits, attention_weights
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get memory attention weights for interpretability."""
        with torch.no_grad():
            _, attention_weights = self.forward(x)
            return attention_weights


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in fraud detection.
    
    Reference: Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal loss for dense object detection. ICCV, 2017.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Logits from model (batch_size,)
            targets: Ground truth labels (batch_size,)
            
        Returns:
            Focal loss value
        """
        # Convert to probabilities
        probs = torch.sigmoid(inputs)
        targets = targets.float()
        
        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Compute p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute alpha weight
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_weight * bce_loss
        else:
            focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss