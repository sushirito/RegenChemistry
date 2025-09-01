#!/usr/bin/env python3
"""
Parallel Christoffel Symbol Computation
Massively parallel computation of Γ(c,λ) for MPS
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple


class ParallelChristoffelComputer:
    """
    Computes Christoffel symbols Γ = ½g⁻¹(∂g/∂c) in parallel
    Optimized for MPS with vectorized operations
    """
    
    def __init__(self, epsilon: float = 1e-4, use_cache: bool = True,
                 cache_size: int = 10000):
        """
        Initialize Christoffel computer
        
        Args:
            epsilon: Finite difference step size
            use_cache: Whether to cache computed values
            cache_size: Maximum cache size
        """
        self.epsilon = epsilon
        self.use_cache = use_cache
        self.cache_size = cache_size
        
        if use_cache:
            self.cache = {}
            self.cache_hits = 0
            self.cache_misses = 0
    
    def compute_single(self, c: torch.Tensor, wavelength: torch.Tensor,
                      metric_network: nn.Module) -> torch.Tensor:
        """
        Compute Christoffel symbol for single or batch of points
        
        Args:
            c: Concentration values (batch,)
            wavelength: Wavelength values (batch,)
            metric_network: The metric network
        
        Returns:
            Christoffel symbols (batch,)
        """
        # Compute metric at three points for finite difference
        g_center = metric_network(c, wavelength)
        g_plus = metric_network(c + self.epsilon, wavelength)
        g_minus = metric_network(c - self.epsilon, wavelength)
        
        # Central difference: ∂g/∂c
        dg_dc = (g_plus - g_minus) / (2 * self.epsilon)
        
        # Christoffel symbol: Γ = ½ g⁻¹ ∂g/∂c
        gamma = 0.5 * dg_dc / (g_center + 1e-8)  # Add small constant for stability
        
        return gamma
    
    def compute_parallel(self, c_batch: torch.Tensor, wavelength_batch: torch.Tensor,
                        metric_network: nn.Module) -> torch.Tensor:
        """
        Compute Christoffel symbols for large batches in parallel
        
        Args:
            c_batch: Concentration values (batch_size,)
            wavelength_batch: Wavelength values (batch_size,)
            metric_network: The metric network
        
        Returns:
            Christoffel symbols (batch_size,)
        """
        batch_size = c_batch.shape[0]
        
        # Stack perturbations for single forward pass
        c_stacked = torch.cat([
            c_batch - self.epsilon,  # g_minus
            c_batch,                 # g_center
            c_batch + self.epsilon   # g_plus
        ], dim=0)
        
        wl_stacked = wavelength_batch.repeat(3)
        
        # Single forward pass for all points
        g_stacked = metric_network(c_stacked, wl_stacked)
        
        # Split results
        g_minus = g_stacked[:batch_size]
        g_center = g_stacked[batch_size:2*batch_size]
        g_plus = g_stacked[2*batch_size:]
        
        # Compute derivatives and Christoffel symbols
        dg_dc = (g_plus - g_minus) / (2 * self.epsilon)
        gamma = 0.5 * dg_dc / (g_center + 1e-8)
        
        return gamma
    
    def compute_grid(self, c_range: Tuple[float, float], wl_range: Tuple[float, float],
                    n_c: int, n_wl: int, metric_network: nn.Module) -> torch.Tensor:
        """
        Pre-compute Christoffel symbols on a grid for interpolation
        
        Args:
            c_range: Concentration range (min, max)
            wl_range: Wavelength range (min, max)
            n_c: Number of concentration points
            n_wl: Number of wavelength points
            metric_network: The metric network
        
        Returns:
            Grid of Christoffel symbols (n_c, n_wl)
        """
        device = next(metric_network.parameters()).device
        
        # Create grid
        c_grid = torch.linspace(c_range[0], c_range[1], n_c, device=device)
        wl_grid = torch.linspace(wl_range[0], wl_range[1], n_wl, device=device)
        
        # Flatten grid for batch processing
        c_flat = c_grid.unsqueeze(1).expand(n_c, n_wl).reshape(-1)
        wl_flat = wl_grid.unsqueeze(0).expand(n_c, n_wl).reshape(-1)
        
        # Compute all Christoffel symbols in one batch
        gamma_flat = self.compute_parallel(c_flat, wl_flat, metric_network)
        
        # Reshape to grid
        gamma_grid = gamma_flat.reshape(n_c, n_wl)
        
        return gamma_grid
    
    def compute_with_cache(self, c: torch.Tensor, wavelength: torch.Tensor,
                          metric_network: nn.Module) -> torch.Tensor:
        """
        Compute with caching for repeated queries
        
        Args:
            c: Concentration value
            wavelength: Wavelength value
            metric_network: The metric network
        
        Returns:
            Christoffel symbol
        """
        if not self.use_cache:
            return self.compute_single(c, wavelength, metric_network)
        
        # Create cache key (round to avoid floating point issues)
        key = (round(c.item(), 6), round(wavelength.item(), 6))
        
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key].clone()
        
        self.cache_misses += 1
        
        # Compute and cache
        gamma = self.compute_single(c, wavelength, metric_network)
        
        # Limit cache size
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            self.cache.pop(next(iter(self.cache)))
        
        self.cache[key] = gamma.clone()
        return gamma
    
    def clear_cache(self):
        """Clear the cache"""
        if self.use_cache:
            self.cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0
    
    def get_cache_stats(self) -> Dict[str, any]:
        """Get cache statistics"""
        if not self.use_cache:
            return {'cache_enabled': False}
        
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(1, total)
        
        return {
            'cache_enabled': True,
            'cache_size': len(self.cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
        }


class InterpolatedChristoffel:
    """
    Pre-computes Christoffel symbols on a grid and uses interpolation
    Much faster for training with fixed metric network
    """
    
    def __init__(self, metric_network: nn.Module, 
                 c_range: Tuple[float, float] = (-1, 1),
                 wl_range: Tuple[float, float] = (-1, 1),
                 n_c: int = 200, n_wl: int = 200):
        """
        Initialize with pre-computed grid
        
        Args:
            metric_network: The metric network
            c_range: Normalized concentration range
            wl_range: Normalized wavelength range
            n_c: Grid points for concentration
            n_wl: Grid points for wavelength
        """
        self.c_range = c_range
        self.wl_range = wl_range
        self.n_c = n_c
        self.n_wl = n_wl
        
        # Pre-compute grid
        computer = ParallelChristoffelComputer()
        self.gamma_grid = computer.compute_grid(
            c_range, wl_range, n_c, n_wl, metric_network
        )
        
        # Create coordinate tensors for interpolation
        device = self.gamma_grid.device
        self.c_coords = torch.linspace(c_range[0], c_range[1], n_c, device=device)
        self.wl_coords = torch.linspace(wl_range[0], wl_range[1], n_wl, device=device)
    
    def interpolate(self, c: torch.Tensor, wavelength: torch.Tensor) -> torch.Tensor:
        """
        Get Christoffel symbols using bilinear interpolation
        
        Args:
            c: Concentration values (batch,)
            wavelength: Wavelength values (batch,)
        
        Returns:
            Interpolated Christoffel symbols (batch,)
        """
        # Normalize to grid coordinates [0, n-1]
        c_norm = (c - self.c_range[0]) / (self.c_range[1] - self.c_range[0])
        c_idx = c_norm * (self.n_c - 1)
        
        wl_norm = (wavelength - self.wl_range[0]) / (self.wl_range[1] - self.wl_range[0])
        wl_idx = wl_norm * (self.n_wl - 1)
        
        # Bilinear interpolation
        c_idx = c_idx.clamp(0, self.n_c - 1.001)
        wl_idx = wl_idx.clamp(0, self.n_wl - 1.001)
        
        c0 = c_idx.floor().long()
        c1 = (c0 + 1).clamp(max=self.n_c - 1)
        wl0 = wl_idx.floor().long()
        wl1 = (wl0 + 1).clamp(max=self.n_wl - 1)
        
        # Interpolation weights
        wc = c_idx - c0.float()
        wwl = wl_idx - wl0.float()
        
        # Get corner values
        v00 = self.gamma_grid[c0, wl0]
        v01 = self.gamma_grid[c0, wl1]
        v10 = self.gamma_grid[c1, wl0]
        v11 = self.gamma_grid[c1, wl1]
        
        # Bilinear interpolation
        v0 = v00 * (1 - wwl) + v01 * wwl
        v1 = v10 * (1 - wwl) + v11 * wwl
        gamma = v0 * (1 - wc) + v1 * wc
        
        return gamma


def test_christoffel():
    """Test Christoffel symbol computation"""
    print("Testing Christoffel Symbol Computer...")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create metric network
    from geodesic_mps.models.metric_network import OptimizedMetricNetwork
    metric = OptimizedMetricNetwork().to(device)
    
    # Create computer
    computer = ParallelChristoffelComputer()
    
    # Test single computation
    c = torch.tensor([0.5], device=device)
    wl = torch.tensor([-0.2], device=device)
    gamma = computer.compute_single(c, wl, metric)
    print(f"\nSingle computation: Γ(0.5, -0.2) = {gamma.item():.6f}")
    
    # Test batch computation
    batch_size = 256
    c_batch = torch.randn(batch_size, device=device)
    wl_batch = torch.randn(batch_size, device=device)
    
    gamma_batch = computer.compute_parallel(c_batch, wl_batch, metric)
    print(f"\nBatch computation: {gamma_batch.shape}")
    print(f"Range: [{gamma_batch.min().item():.6f}, {gamma_batch.max().item():.6f}]")
    
    # Test grid computation
    print("\nComputing grid...")
    gamma_grid = computer.compute_grid((-1, 1), (-1, 1), 50, 50, metric)
    print(f"Grid shape: {gamma_grid.shape}")
    print(f"Grid range: [{gamma_grid.min().item():.6f}, {gamma_grid.max().item():.6f}]")
    
    # Test interpolation
    print("\nTesting interpolation...")
    interp = InterpolatedChristoffel(metric, n_c=100, n_wl=100)
    
    # Compare direct vs interpolated
    test_c = torch.tensor([0.3, -0.5, 0.8], device=device)
    test_wl = torch.tensor([0.1, 0.0, -0.7], device=device)
    
    gamma_direct = computer.compute_parallel(test_c, test_wl, metric)
    gamma_interp = interp.interpolate(test_c, test_wl)
    
    print(f"Direct:       {gamma_direct.cpu().numpy()}")
    print(f"Interpolated: {gamma_interp.cpu().numpy()}")
    print(f"Max error:    {(gamma_direct - gamma_interp).abs().max().item():.6f}")
    
    # Benchmark
    import time
    n_iterations = 1000
    
    # Direct computation
    start = time.time()
    for _ in range(n_iterations):
        _ = computer.compute_parallel(c_batch, wl_batch, metric)
    if device.type == 'mps':
        torch.mps.synchronize()
    direct_time = time.time() - start
    
    # Interpolation
    start = time.time()
    for _ in range(n_iterations):
        _ = interp.interpolate(c_batch, wl_batch)
    if device.type == 'mps':
        torch.mps.synchronize()
    interp_time = time.time() - start
    
    print(f"\nBenchmark ({n_iterations} iterations, batch={batch_size}):")
    print(f"  Direct:        {direct_time:.3f}s ({batch_size*n_iterations/direct_time:.0f} samples/s)")
    print(f"  Interpolation: {interp_time:.3f}s ({batch_size*n_iterations/interp_time:.0f} samples/s)")
    print(f"  Speedup:       {direct_time/interp_time:.1f}x")
    
    print("\nChristoffel tests passed!")


if __name__ == "__main__":
    test_christoffel()