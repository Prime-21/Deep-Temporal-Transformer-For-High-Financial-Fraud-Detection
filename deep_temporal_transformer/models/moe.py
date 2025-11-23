"""
Mixture of Experts (MoE) Layer for Adaptive Fraud Detection

Implements sparse mixture of experts routing where transactions are dynamically
routed to specialized expert networks based on learned patterns.

This allows the model to develop specialized sub-networks for different
fraud patterns (e.g., velocity attacks, account takeover, unusual spending).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math


class Expert(nn.Module):
    """
    Single expert network (feed-forward network).
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SparseRouter(nn.Module):
    """
    Sparse gating network for routing inputs to experts.
    
    Routes each input to top-k experts using learned gating weights.
    Includes noise for load balancing during training.
    
    Args:
        input_dim: Input dimension
        num_experts: Number of expert networks
        top_k: Number of experts to route to per input
        noise_std: Standard deviation of noise for load balancing
    """
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int = 2,
        noise_std: float = 0.1
    ):
        super().__init__()
        assert top_k <= num_experts
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts * 2),
            nn.GELU(),
            nn.Linear(num_experts * 2, num_experts)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, input_dim) or (batch_size, input_dim)
            
        Returns:
            gates: (batch_size, seq_len, num_experts) - normalized gating weights
            expert_indices: (batch_size, seq_len, top_k) - selected expert indices
            load_balance_loss: Scalar - auxiliary loss for load balancing
        """
        original_shape = x.shape
        if x.dim() == 3:
            batch_size, seq_len, _ = x.shape
            x = x.view(-1, x.shape[-1])  # Flatten to (batch * seq_len, input_dim)
        else:
            batch_size = x.shape[0]
            seq_len = 1
        
        # Compute gating logits
        gate_logits = self.gate(x)  # (batch * seq_len, num_experts)
        
        # Add noise during training for exploration and load balancing
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise
        
        # Select top-k experts
        gates_full = F.softmax(gate_logits, dim=-1)
        top_k_gates, expert_indices = torch.topk(gates_full, self.top_k, dim=-1)
        
        # Renormalize top-k gates
        top_k_gates = top_k_gates / top_k_gates.sum(dim=-1, keepdim=True)
        
        # Load balancing loss (encourage uniform expert usage)
        # Importance: average gate value per expert
        importance = gates_full.mean(dim=0)
        # Load: frequency of expert being in top-k
        # expert_indices shape: (batch * seq_len, top_k)
        # Create one-hot encoding for each expert
        batch_size_flat = expert_indices.shape[0]
        expert_mask = torch.zeros(batch_size_flat, self.num_experts, device=x.device)
        for k in range(self.top_k):
            expert_mask.scatter_(1, expert_indices[:, k:k+1], 1.0)
        load = expert_mask.mean(dim=0)  # (num_experts,)
        
        # Coefficient of variation loss
        load_balance_loss = self.num_experts * (importance * load).sum()
        
        # Reshape back if needed
        if len(original_shape) == 3:
            top_k_gates = top_k_gates.view(batch_size, seq_len, self.top_k)
            expert_indices = expert_indices.view(batch_size, seq_len, self.top_k)
        
        return top_k_gates, expert_indices, load_balance_loss


class MixtureOfExperts(nn.Module):
    """
    Sparse Mixture of Experts layer.
    
    Routes each input to a subset of specialized expert networks,
    combining their outputs via learned gating weights.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Expert hidden dimension
        output_dim: Output dimension
        num_experts: Number of expert networks
        top_k: Number of experts to activate per input
        dropout: Dropout probability
        load_balance_weight: Weight for load balancing auxiliary loss
        
    References:
        - "Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer"
          (Shazeer et al., 2017)
        - "Switch Transformers: Scaling to Trillion Parameter Models" (Fedus et al., 2021)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: Optional[int] = None,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        load_balance_weight: float = 0.01
    ):
        super().__init__()
        
        output_dim = output_dim or input_dim
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight
        
        # Create expert networks
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim, dropout)
            for _ in range(num_experts)
        ])
        
        # Gating / routing network
        self.router = SparseRouter(input_dim, num_experts, top_k)
        
        # Track expert usage for monitoring
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        
    def forward(
        self,
        x: torch.Tensor,
        return_aux_loss: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch_size, seq_len, input_dim) or (batch_size, input_dim)
            return_aux_loss: Whether to return auxiliary load balancing loss
            
        Returns:
            output: (batch_size, seq_len, output_dim) - combined expert outputs
            aux_loss: Optional scalar - load balancing loss
        """
        original_shape = x.shape
        seq_mode = x.dim() == 3
        
        if seq_mode:
            batch_size, seq_len, _ = x.shape
            x_flat = x.view(-1, x.shape[-1])  # (batch * seq_len, input_dim)
        else:
            x_flat = x
            batch_size, seq_len = x.shape[0], 1
        
        # Route to experts
        top_k_gates, expert_indices, load_balance_loss = self.router(x_flat)
        # top_k_gates: (batch * seq_len, top_k)
        # expert_indices: (batch * seq_len, top_k)
        
        # Process through experts
        # Initialize output
        output = torch.zeros(
            x_flat.shape[0], self.output_dim,
            device=x.device, dtype=x.dtype
        )
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find which inputs are routed to this expert
            mask = (expert_indices == expert_idx).any(dim=-1)  # (batch * seq_len,)
            
            if mask.any():
                # Get inputs for this expert
                expert_inputs = x_flat[mask]  # (num_routed, input_dim)
                
                # Get corresponding gates
                # Find position of this expert in top-k
                expert_pos = (expert_indices[mask] == expert_idx).long().argmax(dim=-1)
                expert_gates = top_k_gates[mask].gather(1, expert_pos.unsqueeze(-1)).squeeze(-1)
                # (num_routed,)
                
                # Process through expert
                expert_output = self.experts[expert_idx](expert_inputs)  # (num_routed, output_dim)
                
                # Weight by gate and accumulate
                output[mask] += expert_gates.unsqueeze(-1) * expert_output
                
                # Update usage statistics (for monitoring)
                if self.training:
                    self.expert_usage[expert_idx] += mask.sum().item()
        
        # Reshape back
        if seq_mode:
            output = output.view(batch_size, seq_len, self.output_dim)
        
        # Auxiliary loss
        aux_loss = None
        if return_aux_loss and self.training:
            aux_loss = self.load_balance_weight * load_balance_loss
        
        return output, aux_loss
    
    def get_expert_usage_stats(self) -> torch.Tensor:
        """Get normalized expert usage statistics."""
        if self.expert_usage.sum() > 0:
            return self.expert_usage / self.expert_usage.sum()
        return self.expert_usage


class HierarchicalMoE(nn.Module):
    """
    Hierarchical Mixture of Experts with two-level routing.
    
    First routes to groups of experts, then routes within each group.
    This allows for more specialized expertise and better scaling.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Expert hidden dimension
        output_dim: Output dimension
        num_groups: Number of expert groups
        experts_per_group: Experts per group
        top_k_groups: Number of groups to route to
        top_k_experts: Number of experts per group to route to
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: Optional[int] = None,
        num_groups: int = 4,
        experts_per_group: int = 4,
        top_k_groups: int = 2,
        top_k_experts: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        output_dim = output_dim or input_dim
        
        self.num_groups = num_groups
        self.experts_per_group = experts_per_group
        self.top_k_groups = top_k_groups
        self.top_k_experts = top_k_experts
        
        # Group-level router
        self.group_router = SparseRouter(input_dim, num_groups, top_k_groups)
        
        # Expert groups (each group is a MoE)
        self.expert_groups = nn.ModuleList([
            MixtureOfExperts(
                input_dim,
                hidden_dim,
                output_dim,
                num_experts=experts_per_group,
                top_k=top_k_experts,
                dropout=dropout
            )
            for _ in range(num_groups)
        ])
        
    def forward(
        self,
        x: torch.Tensor,
        return_aux_loss: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            return_aux_loss: Whether to return auxiliary losses
            
        Returns:
            output: (batch_size, seq_len, output_dim)
            aux_loss: Optional scalar - combined auxiliary losses
        """
        # First-level routing to groups
        group_gates, group_indices, group_load_loss = self.group_router(x)
        
        # Initialize output
        output = torch.zeros_like(x)
        total_aux_loss = group_load_loss
        
        # Process through selected groups
        for group_idx in range(self.num_groups):
            # Check if this group is selected for any inputs
            mask = (group_indices == group_idx).any(dim=-1)  # (batch, seq_len)
            
            if mask.any():
                # Get group gate weights
                group_pos = (group_indices[mask] == group_idx).long().argmax(dim=-1)
                gate_weights = group_gates[mask].gather(1, group_pos.unsqueeze(-1)).squeeze(-1)
                
                # Process through group MoE
                group_input = x[mask]
                group_output, expert_aux_loss = self.expert_groups[group_idx](
                    group_input,
                    return_aux_loss=return_aux_loss
                )
                
                # Accumulate weighted output
                output[mask] += gate_weights.unsqueeze(-1) * group_output
                
                # Accumulate auxiliary loss
                if expert_aux_loss is not None:
                    total_aux_loss += expert_aux_loss
        
        aux_loss = total_aux_loss if return_aux_loss and self.training else None
        
        return output, aux_loss


# Export public APIs
__all__ = [
    'Expert',
    'SparseRouter',
    'MixtureOfExperts',
    'HierarchicalMoE'
]
