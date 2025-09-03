"""
Riemannian metric network g(c,λ)
Shared across all 6 models in leave-one-out validation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MetricNetwork(nn.Module):
    """Neural network learning the Riemannian metric of spectral space"""
    
    def __init__(self, 
                 input_dim: int = 2,
                 hidden_dims: list = None,
                 activation: str = 'tanh',
                 use_batch_norm: bool = False):
        """
        Initialize metric network
        
        Args:
            input_dim: Input dimension [c, λ]
            hidden_dims: Hidden layer dimensions (must be multiples of 8 for Tensor Cores)
            activation: Activation function
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 256]  # Tensor Core optimized
            
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.use_batch_norm = use_batch_norm
        
        # Build network layers
        layers = []
        in_features = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(in_features, hidden_dim))
            
            # Batch norm (optional)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
                
            # Activation
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
                
            in_features = hidden_dim
            
        # Output layer (no activation, will apply softplus for positivity)
        layers.append(nn.Linear(in_features, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights for stability
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing metric value
        
        Args:
            inputs: [batch_size, 2] containing [c, λ]
            
        Returns:
            g(c,λ): Positive metric values [batch_size, 1]
        """
        # Ensure inputs are properly shaped
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
            
        # Forward through network
        raw_output = self.network(inputs)
        
        # Apply softplus to ensure positivity with minimum value
        # g(c,λ) = softplus(raw) + 0.1
        # This ensures g > 0.1 for numerical stability
        metric = F.softplus(raw_output) + 0.1
        
        return metric
        
    def compute_derivatives(self, c: torch.Tensor, wavelength: torch.Tensor,
                          create_graph: bool = False) -> dict:
        """
        Compute metric and its derivatives
        
        Args:
            c: Concentration values [batch_size]
            wavelength: Wavelength values [batch_size]
            create_graph: Whether to create computation graph for higher derivatives
            
        Returns:
            Dictionary with metric value and derivatives
        """
        # Enable gradients
        c = c.requires_grad_(True)
        
        # Stack inputs
        inputs = torch.stack([c, wavelength], dim=1)
        
        # Compute metric
        g = self.forward(inputs)
        
        # Compute first derivative dg/dc
        dg_dc = torch.autograd.grad(
            outputs=g.sum(),
            inputs=c,
            create_graph=create_graph,
            retain_graph=True
        )[0]
        
        results = {
            'g': g.squeeze(),
            'dg_dc': dg_dc
        }
        
        # Optionally compute second derivative
        if create_graph:
            d2g_dc2 = torch.autograd.grad(
                outputs=dg_dc.sum(),
                inputs=c,
                create_graph=True,
                retain_graph=True
            )[0]
            results['d2g_dc2'] = d2g_dc2
            
        return results
        
    def get_smoothness_loss(self, c_batch: torch.Tensor, 
                          wavelength_batch: torch.Tensor) -> torch.Tensor:
        """
        Compute smoothness regularization loss
        Penalizes rapid changes in metric curvature
        
        Args:
            c_batch: Concentration values [batch_size]
            wavelength_batch: Wavelength values [batch_size]
            
        Returns:
            Smoothness loss scalar
        """
        # Compute second derivatives
        derivatives = self.compute_derivatives(c_batch, wavelength_batch, create_graph=True)
        
        # L2 penalty on second derivative
        if 'd2g_dc2' in derivatives:
            smoothness = (derivatives['d2g_dc2'] ** 2).mean()
        else:
            smoothness = torch.tensor(0.0, device=c_batch.device)
            
        return smoothness
        
    def get_bounds_loss(self, c_batch: torch.Tensor,
                       wavelength_batch: torch.Tensor,
                       min_val: float = 0.01,
                       max_val: float = 100.0) -> torch.Tensor:
        """
        Compute bounds regularization loss
        Keeps metric values in reasonable range
        
        Args:
            c_batch: Concentration values [batch_size]
            wavelength_batch: Wavelength values [batch_size]
            min_val: Minimum allowed metric value
            max_val: Maximum allowed metric value
            
        Returns:
            Bounds loss scalar
        """
        inputs = torch.stack([c_batch, wavelength_batch], dim=1)
        g = self.forward(inputs).squeeze()
        
        # Penalize values outside bounds
        lower_violation = F.relu(min_val - g)
        upper_violation = F.relu(g - max_val)
        
        bounds_loss = lower_violation.mean() + upper_violation.mean()
        
        return bounds_loss