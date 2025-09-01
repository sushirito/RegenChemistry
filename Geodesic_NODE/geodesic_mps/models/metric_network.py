#!/usr/bin/env python3
"""
Metric Network with MPS Optimizations
Ultra-small network to learn Riemannian metric g(c,λ)
"""

import torch
import torch.nn as nn
from typing import Optional


class MetricNetwork(nn.Module):
    """
    Ultra-small network for learning the Riemannian metric
    Optimized for MPS with batch processing
    """
    
    def __init__(self, hidden_dim: int = 8, n_layers: int = 2):
        """
        Initialize metric network
        
        Args:
            hidden_dim: Hidden layer dimension (small to prevent overfitting)
            n_layers: Number of hidden layers
        """
        super().__init__()
        
        # Ultra-small architecture: 2 → 8 → 8 → 1
        layers = []
        
        # Input layer
        layers.append(nn.Linear(2, hidden_dim))
        layers.append(nn.Softplus(beta=2))  # Smooth activation
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Softplus(beta=2))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights for stability
        self._initialize_weights()
        
        # Positive offset to ensure metric is always positive
        self.register_buffer('offset', torch.tensor(0.1))
        
    def _initialize_weights(self):
        """Initialize weights for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, c: torch.Tensor, wavelength: torch.Tensor) -> torch.Tensor:
        """
        Compute metric g(c,λ)
        
        Args:
            c: Normalized concentration values (batch_size,) or (batch,)
            wavelength: Normalized wavelength values (batch_size,) or (batch,)
        
        Returns:
            Metric values (batch_size,), always positive
        """
        # Ensure inputs are the same shape
        if c.dim() == 0:
            c = c.unsqueeze(0)
        if wavelength.dim() == 0:
            wavelength = wavelength.unsqueeze(0)
        
        # Stack inputs
        x = torch.stack([c, wavelength], dim=-1)  # (batch, 2)
        
        # Forward pass
        g = self.network(x).squeeze(-1)  # (batch,)
        
        # Ensure positive metric with softplus
        g = torch.nn.functional.softplus(g, beta=2) + self.offset
        
        return g
    
    def forward_batch(self, c: torch.Tensor, wavelengths: torch.Tensor) -> torch.Tensor:
        """
        Optimized batch forward for multiple wavelengths
        
        Args:
            c: Concentration values (batch_size,)
            wavelengths: Wavelength values (batch_size, n_wavelengths)
        
        Returns:
            Metric values (batch_size, n_wavelengths)
        """
        batch_size, n_wavelengths = wavelengths.shape
        
        # Expand concentration for all wavelengths
        c_expanded = c.unsqueeze(1).expand(-1, n_wavelengths)  # (batch, n_wl)
        
        # Flatten for single forward pass
        c_flat = c_expanded.reshape(-1)  # (batch * n_wl,)
        wl_flat = wavelengths.reshape(-1)  # (batch * n_wl,)
        
        # Compute metric
        g_flat = self.forward(c_flat, wl_flat)  # (batch * n_wl,)
        
        # Reshape back
        g = g_flat.reshape(batch_size, n_wavelengths)
        
        return g
    
    def compute_derivative(self, c: torch.Tensor, wavelength: torch.Tensor,
                         epsilon: float = 1e-4) -> torch.Tensor:
        """
        Compute ∂g/∂c using finite differences
        
        Args:
            c: Concentration values
            wavelength: Wavelength values
            epsilon: Finite difference step
        
        Returns:
            Derivative values
        """
        # Compute g at three points
        g_plus = self.forward(c + epsilon, wavelength)
        g_minus = self.forward(c - epsilon, wavelength)
        
        # Central difference
        dg_dc = (g_plus - g_minus) / (2 * epsilon)
        
        return dg_dc
    
    def regularization_loss(self) -> torch.Tensor:
        """
        Compute regularization loss to ensure smooth metric
        
        Returns:
            Regularization loss scalar
        """
        # L2 regularization on weights
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)
        
        return 1e-4 * l2_loss


class OptimizedMetricNetwork(MetricNetwork):
    """
    MPS-optimized version with additional performance features
    """
    
    def __init__(self, hidden_dim: int = 8, n_layers: int = 2,
                 use_compile: bool = True):
        super().__init__(hidden_dim, n_layers)
        
        # Try to compile the network for better performance
        if use_compile and hasattr(torch, 'compile'):
            try:
                self.network = torch.compile(self.network, backend='aot_eager')
            except:
                pass  # Compilation not available or failed
    
    def forward_parallel(self, c_batch: torch.Tensor, 
                        wavelength_batch: torch.Tensor) -> torch.Tensor:
        """
        Massively parallel forward pass for all combinations
        
        Args:
            c_batch: Concentration values (n_concentrations,)
            wavelength_batch: Wavelength values (n_wavelengths,)
        
        Returns:
            Metric grid (n_concentrations, n_wavelengths)
        """
        n_c = len(c_batch)
        n_wl = len(wavelength_batch)
        
        # Create grid of all combinations
        c_grid = c_batch.unsqueeze(1).expand(n_c, n_wl)
        wl_grid = wavelength_batch.unsqueeze(0).expand(n_c, n_wl)
        
        # Flatten
        c_flat = c_grid.reshape(-1)
        wl_flat = wl_grid.reshape(-1)
        
        # Single forward pass
        g_flat = self.forward(c_flat, wl_flat)
        
        # Reshape to grid
        g_grid = g_flat.reshape(n_c, n_wl)
        
        return g_grid


def test_metric_network():
    """Test metric network functionality"""
    print("Testing Metric Network...")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create network
    metric = OptimizedMetricNetwork().to(device)
    print(f"Network created with {sum(p.numel() for p in metric.parameters())} parameters")
    
    # Test single forward
    c = torch.tensor([0.5], device=device)
    wl = torch.tensor([-0.2], device=device)
    g = metric(c, wl)
    print(f"\nSingle forward: g(0.5, -0.2) = {g.item():.4f}")
    
    # Test batch forward
    batch_size = 32
    c_batch = torch.randn(batch_size, device=device)
    wl_batch = torch.randn(batch_size, device=device)
    g_batch = metric(c_batch, wl_batch)
    print(f"\nBatch forward: input {c_batch.shape} → output {g_batch.shape}")
    print(f"All positive: {torch.all(g_batch > 0).item()}")
    
    # Test parallel grid computation
    n_c, n_wl = 6, 100
    c_grid = torch.linspace(-1, 1, n_c, device=device)
    wl_grid = torch.linspace(-1, 1, n_wl, device=device)
    g_grid = metric.forward_parallel(c_grid, wl_grid)
    print(f"\nParallel grid: {g_grid.shape}")
    print(f"Min: {g_grid.min().item():.4f}, Max: {g_grid.max().item():.4f}")
    
    # Test gradient flow
    c_test = torch.tensor([0.0], device=device, requires_grad=True)
    wl_test = torch.tensor([0.0], device=device)
    g_test = metric(c_test, wl_test)
    loss = g_test.sum()
    loss.backward()
    print(f"\nGradient test: ∂g/∂c exists = {c_test.grad is not None}")
    
    # Benchmark
    import time
    n_iterations = 1000
    batch_size = 256
    
    c_bench = torch.randn(batch_size, device=device)
    wl_bench = torch.randn(batch_size, device=device)
    
    start = time.time()
    for _ in range(n_iterations):
        _ = metric(c_bench, wl_bench)
    
    if device.type == 'mps':
        torch.mps.synchronize()
    
    elapsed = time.time() - start
    throughput = n_iterations * batch_size / elapsed
    
    print(f"\nBenchmark:")
    print(f"  Time for {n_iterations} iterations: {elapsed:.3f}s")
    print(f"  Throughput: {throughput:.0f} samples/sec")
    
    print("\nMetric network tests passed!")


if __name__ == "__main__":
    test_metric_network()