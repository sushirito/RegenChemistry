#!/usr/bin/env python3
"""
Coupled Geodesic-Spectral ODE System
Integrates position, velocity, and absorbance together
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Callable
from torchdiffeq import odeint, odeint_adjoint


class CoupledGeodesicODE:
    """
    Coupled ODE system evolving [c, v, A] together
    
    Dynamics:
        dc/dt = v
        dv/dt = -Γ(c,λ)v²
        dA/dt = f(c, v, λ) from spectral flow network
    """
    
    def __init__(self, 
                 christoffel_computer: Callable,
                 metric_network: nn.Module,
                 spectral_flow_network: nn.Module):
        """
        Initialize coupled ODE system
        
        Args:
            christoffel_computer: Function to compute Christoffel symbols
            metric_network: Neural network for metric g(c,λ)
            spectral_flow_network: Neural network for dA/dt
        """
        self.christoffel_computer = christoffel_computer
        self.metric_network = metric_network
        self.spectral_flow_network = spectral_flow_network
    
    def dynamics(self, t: torch.Tensor, state: torch.Tensor,
                wavelengths: torch.Tensor) -> torch.Tensor:
        """
        Compute coupled ODE dynamics
        
        State vector: [c, v, A]
        
        Args:
            t: Time (scalar, unused but required)
            state: State tensor (batch*wavelength, 3) containing [c, v, A]
            wavelengths: Wavelength values (batch*wavelength,)
        
        Returns:
            Derivatives [dc/dt, dv/dt, dA/dt] with shape (batch*wavelength, 3)
        """
        # Extract state components
        c = state[:, 0]  # concentration
        v = state[:, 1]  # velocity
        A = state[:, 2]  # absorbance (current value, not used in dynamics)
        
        # Compute Christoffel symbols
        gamma = self.christoffel_computer(c, wavelengths, self.metric_network)
        
        # Compute derivatives
        dc_dt = v
        dv_dt = -gamma * v * v  # Geodesic equation
        dA_dt = self.spectral_flow_network(c, v, wavelengths)  # Spectral dynamics
        
        # Stack derivatives
        derivatives = torch.stack([dc_dt, dv_dt, dA_dt], dim=1)
        
        return derivatives
    
    def integrate(self, initial_state: torch.Tensor, wavelengths: torch.Tensor,
                 t_span: torch.Tensor, method: str = 'dopri5',
                 rtol: float = 1e-5, atol: float = 1e-7,
                 use_adjoint: bool = True) -> torch.Tensor:
        """
        Integrate coupled ODE system
        
        Args:
            initial_state: Initial [c₀, v₀, A₀] (batch*wavelength, 3)
            wavelengths: Wavelength values (batch*wavelength,)
            t_span: Time points for integration
            method: ODE solver method
            rtol: Relative tolerance
            atol: Absolute tolerance
            use_adjoint: Use adjoint method for memory efficiency
        
        Returns:
            Trajectories (n_time_points, batch*wavelength, 3)
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
                       A_source: torch.Tensor, wavelengths: torch.Tensor,
                       n_steps: int = 11, t_final: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Batch integration of coupled system
        
        Args:
            c_source: Source concentrations (batch*wavelength,)
            v_initial: Initial velocities (batch*wavelength,)
            A_source: Source absorbances (batch*wavelength,)
            wavelengths: Wavelengths (batch*wavelength,)
            n_steps: Number of time steps
            t_final: Final time
        
        Returns:
            Dictionary containing:
                - trajectories: Full paths (n_steps, batch*wavelength, 3)
                - c_final: Final concentrations
                - v_final: Final velocities
                - A_final: Final absorbances (this is our prediction!)
        """
        # Create initial state
        initial_state = torch.stack([c_source, v_initial, A_source], dim=1)
        
        # Time points
        t_span = torch.linspace(0, t_final, n_steps, device=c_source.device)
        
        # Integrate
        trajectories = self.integrate(
            initial_state, wavelengths, t_span,
            use_adjoint=True
        )
        
        # Extract final state
        final_state = trajectories[-1]  # (batch*wavelength, 3)
        c_final = final_state[:, 0]
        v_final = final_state[:, 1]
        A_final = final_state[:, 2]
        
        # Extract trajectory components for analysis
        c_trajectory = trajectories[:, :, 0]  # (n_steps, batch*wl)
        v_trajectory = trajectories[:, :, 1]
        A_trajectory = trajectories[:, :, 2]
        
        return {
            'trajectories': trajectories,
            'c_final': c_final,
            'v_final': v_final,
            'A_final': A_final,  # This is our prediction!
            'c_trajectory': c_trajectory,
            'v_trajectory': v_trajectory,
            'A_trajectory': A_trajectory,
        }


class ParallelCoupledODE(CoupledGeodesicODE):
    """
    Optimized version for massive parallel processing
    """
    
    def __init__(self, christoffel_computer: Callable,
                 metric_network: nn.Module,
                 spectral_flow_network: nn.Module,
                 cache_christoffel: bool = False):
        super().__init__(christoffel_computer, metric_network, spectral_flow_network)
        self.cache_christoffel = cache_christoffel
        
        if cache_christoffel:
            from geodesic_mps.core.christoffel import InterpolatedChristoffel
            self.christoffel_interp = InterpolatedChristoffel(
                metric_network, n_c=200, n_wl=200
            )
    
    def dynamics_vectorized(self, t: torch.Tensor, state: torch.Tensor,
                           wavelengths: torch.Tensor) -> torch.Tensor:
        """
        Vectorized dynamics with optional caching
        """
        c = state[:, 0]
        v = state[:, 1]
        
        # Use cached Christoffel if available
        if self.cache_christoffel:
            gamma = self.christoffel_interp.interpolate(c, wavelengths)
        else:
            gamma = self.christoffel_computer(c, wavelengths, self.metric_network)
        
        # Vectorized computation
        derivatives = torch.empty_like(state)
        derivatives[:, 0] = v
        derivatives[:, 1] = -gamma * v.square()
        derivatives[:, 2] = self.spectral_flow_network(c, v, wavelengths)
        
        return derivatives
    
    def integrate_mega_batch(self, c_source: torch.Tensor, v_initial: torch.Tensor,
                            A_source: torch.Tensor, wavelengths: torch.Tensor,
                            micro_batch_size: int = 256) -> Dict[str, torch.Tensor]:
        """
        Process mega-batch with micro-batching
        
        Args:
            c_source: Source concentrations (mega_batch*wavelength,)
            v_initial: Initial velocities (mega_batch*wavelength,)
            A_source: Source absorbances (mega_batch*wavelength,)
            wavelengths: Wavelengths (mega_batch*wavelength,)
            micro_batch_size: Size of micro-batches
        
        Returns:
            Combined results for entire mega-batch
        """
        total_size = c_source.shape[0]
        n_micro = (total_size + micro_batch_size - 1) // micro_batch_size
        
        results = []
        
        for i in range(n_micro):
            start = i * micro_batch_size
            end = min((i + 1) * micro_batch_size, total_size)
            
            micro_result = self.integrate_batch(
                c_source[start:end],
                v_initial[start:end],
                A_source[start:end],
                wavelengths[start:end],
                n_steps=5  # Fewer steps for speed
            )
            
            results.append(micro_result['A_final'])
        
        return torch.cat(results, dim=0)


def test_coupled_ode():
    """Test coupled ODE system"""
    print("Testing Coupled Geodesic-Spectral ODE...")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Setup networks
    from geodesic_mps.models.metric_network import OptimizedMetricNetwork
    from geodesic_mps.models.spectral_flow_network import ParallelSpectralFlow
    from geodesic_mps.core.christoffel import ParallelChristoffelComputer
    
    metric = OptimizedMetricNetwork().to(device)
    spectral_flow = ParallelSpectralFlow().to(device)
    christoffel = ParallelChristoffelComputer()
    
    # Create coupled ODE system
    coupled_ode = CoupledGeodesicODE(
        christoffel.compute_parallel,
        metric,
        spectral_flow
    )
    
    # Test single trajectory
    print("\nTesting single trajectory...")
    c_source = torch.tensor([0.0], device=device)
    v_initial = torch.tensor([1.0], device=device)
    A_source = torch.tensor([0.5], device=device)
    wavelength = torch.tensor([0.0], device=device)
    
    result = coupled_ode.integrate_batch(
        c_source, v_initial, A_source, wavelength,
        n_steps=11
    )
    
    print(f"Initial state: c={c_source.item():.2f}, v={v_initial.item():.2f}, A={A_source.item():.2f}")
    print(f"Final state: c={result['c_final'].item():.2f}, A={result['A_final'].item():.2f}")
    print(f"Trajectory shape: {result['trajectories'].shape}")
    
    # Test batch integration
    print("\nTesting batch integration...")
    batch_size = 256
    c_batch = torch.randn(batch_size, device=device) * 0.5
    v_batch = torch.randn(batch_size, device=device)
    A_batch = torch.randn(batch_size, device=device) * 0.1
    wl_batch = torch.randn(batch_size, device=device)
    
    result_batch = coupled_ode.integrate_batch(
        c_batch, v_batch, A_batch, wl_batch,
        n_steps=11
    )
    
    print(f"Batch shape: {result_batch['trajectories'].shape}")
    print(f"Final absorbances shape: {result_batch['A_final'].shape}")
    print(f"Mean final absorbance: {result_batch['A_final'].mean().item():.4f}")
    
    # Test parallel version
    print("\nTesting parallel coupled ODE...")
    parallel_ode = ParallelCoupledODE(
        christoffel.compute_parallel,
        metric,
        spectral_flow,
        cache_christoffel=True
    )
    
    result_parallel = parallel_ode.integrate_batch(
        c_batch, v_batch, A_batch, wl_batch,
        n_steps=5
    )
    
    print(f"Parallel final absorbances: {result_parallel['A_final'].shape}")
    
    # Benchmark
    print("\nBenchmarking...")
    import time
    
    mega_batch = 1024
    n_wavelengths = 100
    total_size = mega_batch * n_wavelengths
    
    c_mega = torch.randn(total_size, device=device) * 0.5
    v_mega = torch.randn(total_size, device=device)
    A_mega = torch.randn(total_size, device=device) * 0.1
    wl_mega = torch.randn(total_size, device=device)
    
    start = time.time()
    A_final_mega = parallel_ode.integrate_mega_batch(
        c_mega, v_mega, A_mega, wl_mega,
        micro_batch_size=512
    )
    
    if device.type == 'mps':
        torch.mps.synchronize()
    
    elapsed = time.time() - start
    throughput = total_size / elapsed
    
    print(f"Mega-batch size: {total_size:,} ODEs")
    print(f"Integration time: {elapsed:.3f}s")
    print(f"Throughput: {throughput:.0f} ODEs/second")
    print(f"Output shape: {A_final_mega.shape}")
    
    print("\nCoupled ODE tests passed!")


if __name__ == "__main__":
    test_coupled_ode()