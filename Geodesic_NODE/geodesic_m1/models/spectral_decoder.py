"""
Spectral decoder for mapping geodesic features to absorbance
Used in multi-model leave-one-out validation
"""

import torch
import torch.nn as nn
from typing import List


class SpectralDecoder(nn.Module):
    """Decodes geodesic path features to final absorbance predictions"""
    
    def __init__(self,
                 input_features: int = 5,
                 hidden_dims: List[int] = None,
                 activation: str = 'tanh',
                 dropout_rate: float = 0.0):
        """
        Initialize spectral decoder
        
        Args:
            input_features: Number of input features from geodesic path
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 128]  # M1-optimized
            
        self.input_features = input_features
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Build decoder network
        layers = []
        in_features = input_features
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(in_features, hidden_dim))
            
            # Activation
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
                
            # Dropout (except last hidden layer)
            if dropout_rate > 0 and i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout_rate))
                
            in_features = hidden_dim
            
        # Output layer (no activation for regression)
        layers.append(nn.Linear(in_features, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder
        
        Args:
            features: Input features [batch_size, input_features]
                     Expected features: [c_final, c_mean, path_length, max_velocity, wavelength]
                     
        Returns:
            absorbance: Predicted absorbance values [batch_size, 1]
        """
        return self.network(features)
        
    def extract_features(self, trajectories: torch.Tensor, 
                        wavelengths: torch.Tensor) -> torch.Tensor:
        """
        Extract features from geodesic trajectories
        
        Args:
            trajectories: Geodesic trajectories [n_time_points, batch_size, 3]
            wavelengths: Wavelength values [batch_size]
            
        Returns:
            features: Extracted features [batch_size, 5]
        """
        batch_size = trajectories.shape[1]
        
        # Extract concentration and velocity trajectories
        c_trajectory = trajectories[:, :, 0]  # [n_time_points, batch_size]
        v_trajectory = trajectories[:, :, 1]  # [n_time_points, batch_size]
        
        # Feature 1: Final concentration
        c_final = c_trajectory[-1, :]  # [batch_size]
        
        # Feature 2: Mean concentration along path
        c_mean = c_trajectory.mean(dim=0)  # [batch_size]
        
        # Feature 3: Path length (integral of |velocity|)
        dt = 1.0 / (trajectories.shape[0] - 1)
        path_length = (v_trajectory.abs() * dt).sum(dim=0)  # [batch_size]
        
        # Feature 4: Maximum velocity magnitude
        max_velocity = v_trajectory.abs().max(dim=0)[0]  # [batch_size]
        
        # Feature 5: Wavelength (normalized)
        wavelength_feature = wavelengths  # [batch_size]
        
        # Stack all features
        features = torch.stack([
            c_final,
            c_mean, 
            path_length,
            max_velocity,
            wavelength_feature
        ], dim=1)  # [batch_size, 5]
        
        return features