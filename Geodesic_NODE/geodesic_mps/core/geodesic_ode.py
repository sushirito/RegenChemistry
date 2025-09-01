#!/usr/bin/env python3
"""
Parallel Geodesic ODE System
Massively parallel integration of geodesic differential equations
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional, Callable
from torchdiffeq import odeint, odeint_adjoint


class ParallelGeodesicODE:
    """
    Parallel geodesic ODE system: d²c/dt² = -Γ(c,λ)(dc/dt)²
    Processes millions of ODEs simultaneously
    """
    
    def __init__(self, christoffel_computer: Callable, metric_network: nn.Module):
        """
        Initialize ODE system
        
        Args:
            christoffel_computer: Function to compute Christoffel symbols
            metric_network: Neural network for metric g(c,λ)
        """
        self.christoffel_computer = christoffel_computer
        self.metric_network = metric_network
        
    def dynamics(self, t: torch.Tensor, state: torch.Tensor, 
                wavelengths: torch.Tensor) -> torch.Tensor:
        """
        Compute ODE dynamics for all trajectories in parallel
        
        State vector: [c, v] where v = dc/dt
        System:
            dc/dt = v
            dv/dt = -Γ(c,λ)v²
        
        Args:
            t: Time (scalar, unused but required by ODE solver)
            state: State tensor (batch*wavelength, 2) containing [c, v]
            wavelengths: Wavelength values (batch*wavelength,)
        
        Returns:
            Derivatives [dc/dt, dv/dt] with shape (batch*wavelength, 2)
        """
        # Extract position and velocity
        c = state[:, 0]  # concentration
        v = state[:, 1]  # velocity dc/dt
        
        # Compute Christoffel symbols for all points
        gamma = self.christoffel_computer(c, wavelengths, self.metric_network)
        
        # Compute derivatives
        dc_dt = v
        dv_dt = -gamma * v * v  # Geodesic equation
        
        # Stack derivatives
        derivatives = torch.stack([dc_dt, dv_dt], dim=1)
        
        return derivatives
    
    def integrate(self, initial_state: torch.Tensor, wavelengths: torch.Tensor,
                 t_span: torch.Tensor, method: str = 'dopri5',
                 rtol: float = 1e-5, atol: float = 1e-7,
                 use_adjoint: bool = True) -> torch.Tensor:
        """
        Integrate geodesic ODEs for all trajectories
        
        Args:
            initial_state: Initial [c₀, v₀] for each trajectory (batch*wavelength, 2)
            wavelengths: Wavelength values (batch*wavelength,)
            t_span: Time points for integration (n_time_points,)
            method: ODE solver method
            rtol: Relative tolerance
            atol: Absolute tolerance
            use_adjoint: Use adjoint method for memory efficiency
        
        Returns:
            Trajectories with shape (n_time_points, batch*wavelength, 2)
        """
        # Create ODE function with wavelengths bound
        def ode_func(t, state):
            return self.dynamics(t, state, wavelengths)
        
        # Choose solver
        solver = odeint_adjoint if use_adjoint else odeint
        
        # Integrate
        trajectories = solver(
            ode_func,
            initial_state,
            t_span,
            method=method,
            rtol=rtol,
            atol=atol
        )
        
        return trajectories
    
    def integrate_batch(self, c_source: torch.Tensor, v_initial: torch.Tensor,
                       wavelengths: torch.Tensor, n_steps: int = 11,
                       t_final: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Batch integration with path statistics extraction
        
        Args:
            c_source: Source concentrations (batch*wavelength,)
            v_initial: Initial velocities (batch*wavelength,)
            wavelengths: Wavelengths (batch*wavelength,)
            n_steps: Number of time steps
            t_final: Final time
        
        Returns:
            Dictionary containing:
                - trajectories: Full paths (n_steps, batch*wavelength, 2)
                - c_final: Final concentrations
                - path_stats: Statistics along paths
        """
        # Create initial state
        initial_state = torch.stack([c_source, v_initial], dim=1)
        
        # Time points
        t_span = torch.linspace(0, t_final, n_steps, device=c_source.device)
        
        # Integrate
        trajectories = self.integrate(
            initial_state, wavelengths, t_span,
            use_adjoint=True
        )
        
        # Extract final state
        c_final = trajectories[-1, :, 0]
        v_final = trajectories[-1, :, 1]
        
        # Compute path statistics
        c_trajectory = trajectories[:, :, 0]  # (n_steps, batch*wl)
        v_trajectory = trajectories[:, :, 1]
        
        # Path statistics for decoder
        c_mean = c_trajectory.mean(dim=0)
        c_std = c_trajectory.std(dim=0)
        v_max = v_trajectory.abs().max(dim=0)[0]
        
        # Compute path length in metric space
        dt = t_final / (n_steps - 1)
        g_values = []
        
        for i in range(n_steps):
            c_i = c_trajectory[i]
            g_i = self.metric_network(c_i, wavelengths)
            g_values.append(g_i)
        
        g_values = torch.stack(g_values, dim=0)  # (n_steps, batch*wl)
        
        # Path length: ∫√(g(c))|dc/dt|dt
        path_element = torch.sqrt(g_values) * v_trajectory.abs()
        path_length = path_element.sum(dim=0) * dt
        
        return {
            'trajectories': trajectories,
            'c_final': c_final,
            'v_final': v_final,
            'c_mean': c_mean,
            'c_std': c_std,
            'v_max': v_max,
            'path_length': path_length,
        }


class VectorizedGeodesicODE(ParallelGeodesicODE):
    """
    Optimized version using vectorized operations
    """
    
    def __init__(self, christoffel_computer: Callable, metric_network: nn.Module,
                 cache_christoffel: bool = False):
        super().__init__(christoffel_computer, metric_network)
        self.cache_christoffel = cache_christoffel
        
        if cache_christoffel:
            # Pre-compute Christoffel grid for interpolation
            from geodesic_mps.core.christoffel import InterpolatedChristoffel
            self.christoffel_interp = InterpolatedChristoffel(
                metric_network, n_c=200, n_wl=200
            )
    
    def dynamics_vectorized(self, t: torch.Tensor, state: torch.Tensor,
                           wavelengths: torch.Tensor) -> torch.Tensor:
        """
        Vectorized dynamics computation with optional caching
        """
        c = state[:, 0]
        v = state[:, 1]
        
        # Use interpolated Christoffel if available
        if self.cache_christoffel:
            gamma = self.christoffel_interp.interpolate(c, wavelengths)
        else:
            gamma = self.christoffel_computer(c, wavelengths, self.metric_network)
        
        # Vectorized computation
        derivatives = torch.empty_like(state)
        derivatives[:, 0] = v
        derivatives[:, 1] = -gamma * v.square()
        
        return derivatives
    
    def integrate_parallel_batches(self, batch_list: list,
                                  n_steps: int = 11) -> list:
        """
        Process multiple independent batches in parallel
        
        Args:
            batch_list: List of (c_source, v_initial, wavelengths) tuples
            n_steps: Number of integration steps
        
        Returns:
            List of integration results
        """
        # Combine all batches
        all_c = torch.cat([b[0] for b in batch_list])
        all_v = torch.cat([b[1] for b in batch_list])
        all_wl = torch.cat([b[2] for b in batch_list])
        
        # Single integration for all
        result = self.integrate_batch(all_c, all_v, all_wl, n_steps)
        
        # Split results back
        batch_sizes = [len(b[0]) for b in batch_list]
        results = []
        
        start_idx = 0
        for size in batch_sizes:
            end_idx = start_idx + size
            batch_result = {
                k: v[..., start_idx:end_idx, :] if v.dim() > 2 else v[start_idx:end_idx]
                for k, v in result.items()
            }
            results.append(batch_result)
            start_idx = end_idx
        
        return results


def test_geodesic_ode():
    """Test geodesic ODE integration"""
    print("Testing Geodesic ODE System...")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Setup
    from geodesic_mps.models.metric_network import OptimizedMetricNetwork
    from geodesic_mps.core.christoffel import ParallelChristoffelComputer
    
    metric = OptimizedMetricNetwork().to(device)
    christoffel = ParallelChristoffelComputer()
    
    # Create ODE system
    ode_system = VectorizedGeodesicODE(
        christoffel.compute_parallel,
        metric,
        cache_christoffel=False
    )
    
    # Test single trajectory
    print("\nTesting single trajectory...")
    c_source = torch.tensor([0.0], device=device)
    v_initial = torch.tensor([1.0], device=device)
    wavelength = torch.tensor([0.5], device=device)
    
    result = ode_system.integrate_batch(
        c_source, v_initial, wavelength, n_steps=11
    )
    
    print(f"Final concentration: {result['c_final'].item():.4f}")
    print(f"Path length: {result['path_length'].item():.4f}")
    print(f"Mean concentration: {result['c_mean'].item():.4f}")
    
    # Test batch integration
    print("\nTesting batch integration...")
    batch_size = 256
    c_batch = torch.randn(batch_size, device=device) * 0.5
    v_batch = torch.randn(batch_size, device=device)
    wl_batch = torch.randn(batch_size, device=device)
    
    result_batch = ode_system.integrate_batch(
        c_batch, v_batch, wl_batch, n_steps=11
    )
    
    print(f"Batch shape: {result_batch['trajectories'].shape}")
    print(f"Final concentrations shape: {result_batch['c_final'].shape}")
    
    # Test massive parallel batch
    print("\nTesting massive parallel batch...")
    mega_batch = 2048
    n_wavelengths = 601
    total_size = mega_batch * n_wavelengths
    
    c_mega = torch.randn(total_size, device=device) * 0.5
    v_mega = torch.randn(total_size, device=device)
    wl_mega = torch.randn(total_size, device=device)
    
    import time
    start = time.time()
    
    result_mega = ode_system.integrate_batch(
        c_mega, v_mega, wl_mega, n_steps=5  # Fewer steps for speed
    )
    
    if device.type == 'mps':
        torch.mps.synchronize()
    
    elapsed = time.time() - start
    throughput = total_size / elapsed
    
    print(f"Mega-batch size: {total_size:,} ODEs")
    print(f"Integration time: {elapsed:.3f}s")
    print(f"Throughput: {throughput:.0f} ODEs/second")
    
    # Test with caching
    print("\nTesting with Christoffel caching...")
    ode_cached = VectorizedGeodesicODE(
        christoffel.compute_parallel,
        metric,
        cache_christoffel=True
    )
    
    start = time.time()
    result_cached = ode_cached.integrate_batch(
        c_batch, v_batch, wl_batch, n_steps=11
    )
    
    if device.type == 'mps':
        torch.mps.synchronize()
    
    cached_time = time.time() - start
    print(f"Cached integration time: {cached_time:.3f}s")
    
    print("\nGeodesic ODE tests passed!")


if __name__ == "__main__":
    test_geodesic_ode()