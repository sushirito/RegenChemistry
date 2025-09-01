#!/usr/bin/env python3
"""
Metric Network for learning the Riemannian metric g(c,λ)
This defines the geometry of the 1D concentration manifold at each wavelength
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MetricNetwork(nn.Module):
    """
    Neural network that learns the Riemannian metric g(c,λ)
    
    The metric defines how "difficult" it is to move through concentration space
    at a given wavelength. Large g means rapid spectral changes, small g means smooth.
    """
    
    def __init__(self):
        super(MetricNetwork, self).__init__()
        
        # Architecture as specified: 2 → 64 → 128 → 1
        self.layers = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier/He initialization for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier initialization for tanh activations
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, c: torch.Tensor, wavelength: torch.Tensor) -> torch.Tensor:
        """
        Compute the metric g(c,λ)
        
        Args:
            c: Normalized concentration values (batch_size,) or (batch_size, 1)
            wavelength: Normalized wavelength values (batch_size,) or (batch_size, 1)
        
        Returns:
            g: Metric values (batch_size,) - always positive
        """
        # Ensure inputs are properly shaped
        if c.dim() == 1:
            c = c.unsqueeze(-1)
        if wavelength.dim() == 1:
            wavelength = wavelength.unsqueeze(-1)
        
        # Concatenate inputs
        x = torch.cat([c, wavelength], dim=-1)
        
        # Pass through network
        raw_metric = self.layers(x).squeeze(-1)
        
        # Apply softplus + offset to ensure g > 0
        # softplus(x) = log(1 + exp(x)) - smooth version of ReLU
        g = F.softplus(raw_metric) + 0.1
        
        return g
    
    def compute_metric_batch(self, c_batch: torch.Tensor, 
                           wavelength_batch: torch.Tensor) -> torch.Tensor:
        """
        Compute metric for a batch of (c, λ) pairs
        
        Args:
            c_batch: Batch of concentrations (batch_size,)
            wavelength_batch: Batch of wavelengths (batch_size,)
        
        Returns:
            Metric values (batch_size,)
        """
        return self.forward(c_batch, wavelength_batch)
    
    def compute_metric_grid(self, c_range: torch.Tensor, 
                           wavelength: float) -> torch.Tensor:
        """
        Compute metric over a range of concentrations for a fixed wavelength
        Useful for visualization
        
        Args:
            c_range: Range of concentration values (n_points,)
            wavelength: Single wavelength value
        
        Returns:
            Metric values over the concentration range (n_points,)
        """
        wavelength_tensor = torch.full_like(c_range, wavelength)
        return self.forward(c_range, wavelength_tensor)


def compute_metric_smoothness_loss(metric_network: MetricNetwork,
                                  c_samples: torch.Tensor,
                                  wavelength_samples: torch.Tensor,
                                  epsilon: float = 1e-3) -> torch.Tensor:
    """
    Compute smoothness regularization loss (∂²g/∂c²)²
    Encourages the metric to be smooth with respect to concentration
    
    Args:
        metric_network: The metric network
        c_samples: Sample concentrations
        wavelength_samples: Sample wavelengths
        epsilon: Finite difference step size
    
    Returns:
        Smoothness loss (scalar)
    """
    # Compute second derivative using finite differences
    g_center = metric_network(c_samples, wavelength_samples)
    g_plus = metric_network(c_samples + epsilon, wavelength_samples)
    g_minus = metric_network(c_samples - epsilon, wavelength_samples)
    
    # Second derivative approximation
    d2g_dc2 = (g_plus - 2 * g_center + g_minus) / (epsilon ** 2)
    
    # Penalize large second derivatives
    smoothness_loss = torch.mean(d2g_dc2 ** 2)
    
    return smoothness_loss


def compute_metric_bounds_loss(metric_network: MetricNetwork,
                              c_samples: torch.Tensor,
                              wavelength_samples: torch.Tensor,
                              min_g: float = 0.01,
                              max_g: float = 100.0) -> torch.Tensor:
    """
    Compute bounds regularization to keep metric in reasonable range
    
    Args:
        metric_network: The metric network
        c_samples: Sample concentrations
        wavelength_samples: Sample wavelengths
        min_g: Minimum allowed metric value
        max_g: Maximum allowed metric value
    
    Returns:
        Bounds loss (scalar)
    """
    g = metric_network(c_samples, wavelength_samples)
    
    # Penalize values outside bounds
    lower_bound_loss = F.relu(min_g - g).mean()
    upper_bound_loss = F.relu(g - max_g).mean()
    
    return lower_bound_loss + upper_bound_loss


if __name__ == "__main__":
    # Test the metric network
    print("Testing Metric Network...")
    
    # Create network
    metric_net = MetricNetwork()
    
    # Test single forward pass
    c = torch.tensor([0.5], dtype=torch.float32)
    wavelength = torch.tensor([-0.2], dtype=torch.float32)
    g = metric_net(c, wavelength)
    print(f"Single metric value: g = {g.item():.4f}")
    
    # Test batch forward pass
    batch_size = 32
    c_batch = torch.randn(batch_size)
    wavelength_batch = torch.randn(batch_size)
    g_batch = metric_net(c_batch, wavelength_batch)
    print(f"Batch metric shape: {g_batch.shape}")
    print(f"Metric range: [{g_batch.min().item():.4f}, {g_batch.max().item():.4f}]")
    
    # Verify all values are positive
    assert torch.all(g_batch > 0), "Metric must be positive everywhere!"
    print("✓ All metric values are positive")
    
    # Test smoothness loss
    smoothness_loss = compute_metric_smoothness_loss(metric_net, c_batch, wavelength_batch)
    print(f"Smoothness loss: {smoothness_loss.item():.6f}")
    
    # Test bounds loss
    bounds_loss = compute_metric_bounds_loss(metric_net, c_batch, wavelength_batch)
    print(f"Bounds loss: {bounds_loss.item():.6f}")
    
    # Test gradient flow
    loss = g_batch.mean()
    loss.backward()
    
    # Check gradients exist
    for name, param in metric_net.named_parameters():
        if param.grad is not None:
            print(f"✓ Gradient exists for {name}: shape {param.grad.shape}")
    
    print("\nMetric Network tests passed!")