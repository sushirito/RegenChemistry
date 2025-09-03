"""
Spectral flow network for coupled ODE dynamics
Models dA/dt = f(c,v,λ) in the geodesic system
Optimized for M1 Mac MPS acceleration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SpectralFlowNetwork(nn.Module):
    """Neural network modeling spectral flow dA/dt along geodesics"""
    
    def __init__(self, 
                 input_dim: int = 3,
                 hidden_dims: list = None,
                 activation: str = 'tanh'):
        """
        Initialize spectral flow network
        
        Args:
            input_dim: Input dimension [c, v, λ]
            hidden_dims: Hidden layer dimensions (M1-optimized)
            activation: Activation function
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 128]  # M1-friendly dimensions
            
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Build network layers
        layers = []
        in_features = input_dim
        
        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(in_features, hidden_dim))
            
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
            
        # Output layer for dA/dt
        layers.append(nn.Linear(in_features, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights for stable ODE dynamics"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Small initialization to prevent explosive dynamics
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing spectral flow rate
        
        Args:
            inputs: [batch_size, 3] containing [c, v, λ]
            
        Returns:
            dA/dt: Rate of absorbance change [batch_size, 1]
        """
        # Ensure inputs are properly shaped
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
            
        # Forward through network
        dA_dt = self.network(inputs)
        
        return dA_dt