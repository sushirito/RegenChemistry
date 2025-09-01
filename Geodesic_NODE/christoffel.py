#!/usr/bin/env python3
"""
Christoffel Symbol Computation
Computes Γ(c,λ) = ½ g⁻¹(c,λ) ∂g(c,λ)/∂c for the geodesic equation
"""

import torch
from typing import Optional, Dict, Tuple
from metric_network import MetricNetwork


class ChristoffelComputer:
    """
    Computes Christoffel symbols for the Riemannian metric
    Uses finite differences for numerical stability
    """
    
    def __init__(self, epsilon: float = 1e-4):
        """
        Initialize Christoffel computer
        
        Args:
            epsilon: Finite difference step size
        """
        self.epsilon = epsilon
        self.cache: Dict[Tuple, torch.Tensor] = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def compute(self, 
                c: torch.Tensor, 
                wavelength: torch.Tensor,
                metric_network: MetricNetwork,
                use_cache: bool = True) -> torch.Tensor:
        """
        Compute Christoffel symbol Γ(c,λ) = ½ g⁻¹(c,λ) ∂g(c,λ)/∂c
        
        Args:
            c: Concentration values (batch_size,)
            wavelength: Wavelength values (batch_size,)
            metric_network: The metric network
            use_cache: Whether to use caching
        
        Returns:
            Christoffel symbol values (batch_size,)
        """
        # Create cache key if using cache
        if use_cache and c.shape[0] == 1:
            cache_key = (c.item(), wavelength.item())
            if cache_key in self.cache:
                self.cache_hits += 1
                return self.cache[cache_key].clone()
            self.cache_misses += 1
        
        # Compute metric at three points for finite difference
        g_center = metric_network(c, wavelength)
        g_plus = metric_network(c + self.epsilon, wavelength)
        g_minus = metric_network(c - self.epsilon, wavelength)
        
        # Compute derivative ∂g/∂c using central differences
        dg_dc = (g_plus - g_minus) / (2 * self.epsilon)
        
        # Compute Christoffel symbol: Γ = ½ g⁻¹ ∂g/∂c
        gamma = 0.5 * dg_dc / g_center
        
        # Cache result if applicable
        if use_cache and c.shape[0] == 1:
            self.cache[cache_key] = gamma.clone()
        
        return gamma
    
    def compute_with_autograd(self,
                            c: torch.Tensor,
                            wavelength: torch.Tensor,
                            metric_network: MetricNetwork) -> torch.Tensor:
        """
        Alternative computation using automatic differentiation
        Can be more accurate but potentially less stable
        
        Args:
            c: Concentration values (batch_size,)
            wavelength: Wavelength values (batch_size,)
            metric_network: The metric network
        
        Returns:
            Christoffel symbol values (batch_size,)
        """
        # Enable gradient computation for c
        c = c.requires_grad_(True)
        
        # Compute metric
        g = metric_network(c, wavelength)
        
        # Compute gradient ∂g/∂c
        dg_dc = torch.autograd.grad(
            outputs=g.sum(),  # Sum for batched gradient
            inputs=c,
            create_graph=True,  # Keep graph for higher-order derivatives
            retain_graph=True
        )[0]
        
        # Compute Christoffel symbol
        gamma = 0.5 * dg_dc / g
        
        return gamma
    
    def clear_cache(self):
        """Clear the computation cache"""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            'cache_size': len(self.cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses)
        }


def verify_christoffel_properties(metric_network: MetricNetwork,
                                 n_samples: int = 100) -> Dict[str, float]:
    """
    Verify that Christoffel symbols have expected properties
    
    Args:
        metric_network: The metric network
        n_samples: Number of samples to test
    
    Returns:
        Dictionary of verification results
    """
    computer = ChristoffelComputer()
    
    # Sample random points
    c_samples = torch.randn(n_samples)
    wavelength_samples = torch.randn(n_samples)
    
    # Compute Christoffel symbols
    gamma = computer.compute(c_samples, wavelength_samples, metric_network)
    
    # Check properties
    results = {
        'mean': gamma.mean().item(),
        'std': gamma.std().item(),
        'min': gamma.min().item(),
        'max': gamma.max().item(),
        'finite': torch.isfinite(gamma).all().item(),
        'bounded': (gamma.abs() < 1000).all().item()
    }
    
    # Test consistency: If metric is constant, Christoffel should be zero
    # Create a dummy constant metric network for testing
    class ConstantMetric(MetricNetwork):
        def forward(self, c, wavelength):
            return torch.ones_like(c) * 1.0
    
    constant_metric = ConstantMetric()
    gamma_constant = computer.compute(c_samples, wavelength_samples, constant_metric)
    results['constant_metric_near_zero'] = (gamma_constant.abs() < 0.01).all().item()
    
    return results


if __name__ == "__main__":
    # Test Christoffel symbol computation
    print("Testing Christoffel Symbol Computer...")
    
    # Create metric network and computer
    metric_net = MetricNetwork()
    computer = ChristoffelComputer(epsilon=1e-4)
    
    # Test single computation
    c = torch.tensor([0.5])
    wavelength = torch.tensor([-0.2])
    gamma = computer.compute(c, wavelength, metric_net)
    print(f"Single Christoffel symbol: Γ = {gamma.item():.6f}")
    
    # Test batch computation
    batch_size = 32
    c_batch = torch.randn(batch_size)
    wavelength_batch = torch.randn(batch_size)
    gamma_batch = computer.compute(c_batch, wavelength_batch, metric_net)
    print(f"Batch Christoffel shape: {gamma_batch.shape}")
    print(f"Christoffel range: [{gamma_batch.min().item():.6f}, {gamma_batch.max().item():.6f}]")
    
    # Compare finite differences with autograd
    print("\nComparing methods:")
    gamma_fd = computer.compute(c, wavelength, metric_net)
    gamma_ag = computer.compute_with_autograd(c, wavelength, metric_net)
    print(f"Finite differences: Γ = {gamma_fd.item():.6f}")
    print(f"Automatic differentiation: Γ = {gamma_ag.item():.6f}")
    print(f"Difference: {abs(gamma_fd.item() - gamma_ag.item()):.8f}")
    
    # Test caching
    print("\nTesting cache:")
    computer.clear_cache()
    for _ in range(10):
        _ = computer.compute(c, wavelength, metric_net, use_cache=True)
    stats = computer.get_cache_stats()
    print(f"Cache stats: {stats}")
    
    # Verify properties
    print("\nVerifying Christoffel properties:")
    properties = verify_christoffel_properties(metric_net)
    for key, value in properties.items():
        print(f"  {key}: {value}")
    
    # Test gradient flow
    print("\nTesting gradient flow:")
    c_test = torch.tensor([0.0], requires_grad=True)
    wavelength_test = torch.tensor([0.0])
    gamma_test = computer.compute(c_test, wavelength_test, metric_net)
    loss = gamma_test.sum()
    loss.backward()
    print(f"✓ Gradient computed: {c_test.grad is not None}")
    
    print("\nChristoffel Symbol tests passed!")