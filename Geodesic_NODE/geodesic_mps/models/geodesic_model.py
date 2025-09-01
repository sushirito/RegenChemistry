#!/usr/bin/env python3
"""
End-to-End Geodesic Spectral Model with Coupled ODE
Integrates metric network, spectral flow, and shooting solver
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import warnings

from .metric_network import OptimizedMetricNetwork
from .spectral_flow_network import ParallelSpectralFlow
from ..core.christoffel import ParallelChristoffelComputer
from ..core.coupled_ode import CoupledGeodesicODE, ParallelCoupledODE
from ..core.shooting_solver import OptimizedShootingSolver


class CoupledShootingSolver:
    """
    Shooting solver adapted for coupled ODE system
    """
    
    def __init__(self, coupled_ode: CoupledGeodesicODE,
                 max_iterations: int = 10,
                 learning_rate: float = 0.5):
        self.coupled_ode = coupled_ode
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
    
    def solve_batch(self, c_source: torch.Tensor, c_target: torch.Tensor,
                   A_source: torch.Tensor, wavelengths: torch.Tensor,
                   n_trajectory_points: int = 11) -> Dict[str, torch.Tensor]:
        """
        Solve BVP for coupled system
        """
        batch_size = c_source.shape[0]
        device = c_source.device
        
        # Initial guess for velocities
        v_initial = c_target - c_source
        
        # Fixed iterations for parallelism
        for iteration in range(self.max_iterations):
            # Integrate with current velocities
            result = self.coupled_ode.integrate_batch(
                c_source, v_initial, A_source, wavelengths,
                n_steps=n_trajectory_points
            )
            
            c_final = result['c_final']
            
            # Compute errors and update velocities
            errors = c_final - c_target
            v_update = -self.learning_rate * errors
            v_initial = v_initial + v_update
        
        # Final integration with optimal velocities
        final_result = self.coupled_ode.integrate_batch(
            c_source, v_initial, A_source, wavelengths,
            n_steps=n_trajectory_points
        )
        
        return final_result


class GeodesicSpectralModel(nn.Module):
    """
    Complete model using coupled ODE for spectral interpolation
    """
    
    def __init__(self,
                 metric_hidden_dim: int = 8,
                 spectral_hidden_dim: int = 16,
                 n_trajectory_points: int = 11,
                 shooting_max_iter: int = 10,
                 use_christoffel_cache: bool = False,
                 device: Optional[torch.device] = None):
        """
        Initialize complete model
        
        Args:
            metric_hidden_dim: Hidden dimension for metric network
            spectral_hidden_dim: Hidden dimension for spectral flow
            n_trajectory_points: Points along geodesic trajectory
            shooting_max_iter: Max iterations for shooting solver
            use_christoffel_cache: Cache Christoffel symbols
            device: Target device
        """
        super().__init__()
        
        self.device = device or torch.device('cpu')
        self.n_trajectory_points = n_trajectory_points
        
        # Neural networks
        self.metric_network = OptimizedMetricNetwork(
            hidden_dim=metric_hidden_dim,
            n_layers=2
        ).to(self.device)
        
        self.spectral_flow_network = ParallelSpectralFlow(
            hidden_dim=spectral_hidden_dim
        ).to(self.device)
        
        # Mathematical components
        self.christoffel_computer = ParallelChristoffelComputer()
        
        # Coupled ODE system
        self.coupled_ode = ParallelCoupledODE(
            self.christoffel_computer.compute_parallel,
            self.metric_network,
            self.spectral_flow_network,
            cache_christoffel=use_christoffel_cache
        )
        
        # Shooting solver
        self.shooting_solver = CoupledShootingSolver(
            self.coupled_ode,
            max_iterations=shooting_max_iter
        )
        
        # Statistics
        self.last_solve_stats = {}
    
    def forward(self, c_source: torch.Tensor, c_target: torch.Tensor,
               wavelengths: torch.Tensor, A_source: Optional[torch.Tensor] = None,
               return_trajectories: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete model
        
        Args:
            c_source: Source concentrations (batch*wavelength,)
            c_target: Target concentrations (batch*wavelength,)
            wavelengths: Wavelength values (batch*wavelength,)
            A_source: Source absorbances (optional, defaults to zeros)
            return_trajectories: Whether to return full trajectories
        
        Returns:
            Dictionary containing:
                - absorbance: Predicted absorbance at target (A_final)
                - trajectories: Full paths (if requested)
        """
        # Ensure inputs are on correct device
        c_source = c_source.to(self.device)
        c_target = c_target.to(self.device)
        wavelengths = wavelengths.to(self.device)
        
        # Initialize source absorbance if not provided
        if A_source is None:
            A_source = torch.zeros_like(c_source)
        else:
            A_source = A_source.to(self.device)
        
        # Solve boundary value problem with coupled ODE
        result = self.shooting_solver.solve_batch(
            c_source, c_target, A_source, wavelengths,
            n_trajectory_points=self.n_trajectory_points
        )
        
        # Extract final absorbance (our prediction!)
        absorbance = result['A_final']
        
        # Store statistics
        self.last_solve_stats = {
            'final_concentration': result['c_final'],
            'final_velocity': result['v_final'],
            'concentration_error': (result['c_final'] - c_target).abs()
        }
        
        output = {
            'absorbance': absorbance,  # This is A_final from the coupled ODE!
        }
        
        if return_trajectories:
            output['trajectories'] = result.get('trajectories')
            output['A_trajectory'] = result.get('A_trajectory')
        
        return output
    
    def forward_batch(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass with batch dictionary input
        
        Args:
            batch_data: Dictionary from data generator
        
        Returns:
            Predicted absorbance values
        """
        # Extract source absorbance if available
        A_source = batch_data.get('source_absorbance', None)
        
        result = self.forward(
            batch_data['source_conc'],
            batch_data['target_conc'],
            batch_data['wavelengths'],
            A_source=A_source
        )
        return result['absorbance']
    
    def get_model_stats(self) -> Dict[str, any]:
        """
        Get model statistics
        """
        total_params = sum(p.numel() for p in self.parameters())
        metric_params = sum(p.numel() for p in self.metric_network.parameters())
        spectral_params = sum(p.numel() for p in self.spectral_flow_network.parameters())
        
        return {
            'total_parameters': total_params,
            'metric_parameters': metric_params,
            'spectral_flow_parameters': spectral_params,
            'device': str(self.device),
            'trajectory_points': self.n_trajectory_points,
            'last_solve_stats': self.last_solve_stats
        }


class ParallelGeodesicModel(GeodesicSpectralModel):
    """
    Optimized version for massive parallel processing
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Pre-compile if available
        if hasattr(torch, 'compile'):
            try:
                self.metric_network = torch.compile(
                    self.metric_network,
                    backend='aot_eager'
                )
                self.spectral_flow_network = torch.compile(
                    self.spectral_flow_network,
                    backend='aot_eager'
                )
            except:
                warnings.warn("Failed to compile networks")
    
    def forward_mega_batch(self, source_conc: torch.Tensor,
                          target_conc: torch.Tensor,
                          wavelengths: torch.Tensor,
                          source_absorbance: Optional[torch.Tensor] = None,
                          micro_batch_size: int = 256) -> torch.Tensor:
        """
        Process mega-batch with micro-batching
        """
        if source_absorbance is None:
            source_absorbance = torch.zeros_like(source_conc)
        
        return self.coupled_ode.integrate_mega_batch(
            source_conc,
            target_conc - source_conc,  # Initial velocity guess
            source_absorbance,
            wavelengths,
            micro_batch_size
        )


def test_geodesic_model():
    """Test complete geodesic model"""
    print("Testing Complete Geodesic Spectral Model...")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    model = GeodesicSpectralModel(
        metric_hidden_dim=8,
        spectral_hidden_dim=16,
        n_trajectory_points=11,
        shooting_max_iter=5,
        device=device
    )
    
    stats = model.get_model_stats()
    print(f"\nModel created:")
    for key, value in stats.items():
        if key != 'last_solve_stats':
            print(f"  {key}: {value}")
    
    # Test single prediction
    print("\nTesting single prediction...")
    c_source = torch.tensor([-0.5], device=device)
    c_target = torch.tensor([0.5], device=device)
    wavelength = torch.tensor([0.0], device=device)
    
    result = model(c_source, c_target, wavelength, return_trajectories=True)
    
    print(f"Predicted absorbance: {result['absorbance'].item():.4f}")
    if result.get('trajectories') is not None:
        print(f"Trajectory shape: {result['trajectories'].shape}")
    
    # Test batch prediction
    print("\nTesting batch prediction...")
    batch_size = 256
    c_source_batch = torch.randn(batch_size, device=device) * 0.5
    c_target_batch = torch.randn(batch_size, device=device) * 0.5
    wl_batch = torch.randn(batch_size, device=device)
    
    result_batch = model(c_source_batch, c_target_batch, wl_batch)
    
    print(f"Batch output shape: {result_batch['absorbance'].shape}")
    print(f"Mean absorbance: {result_batch['absorbance'].mean().item():.4f}")
    print(f"Std absorbance: {result_batch['absorbance'].std().item():.4f}")
    
    # Test gradient flow
    print("\nTesting gradient flow...")
    loss = result_batch['absorbance'].mean()
    loss.backward()
    
    has_grad = all(
        p.grad is not None 
        for p in model.parameters() 
        if p.requires_grad
    )
    print(f"All parameters have gradients: {has_grad}")
    
    # Test parallel model
    print("\nTesting parallel model...")
    parallel_model = ParallelGeodesicModel(
        metric_hidden_dim=8,
        spectral_hidden_dim=11,
        n_trajectory_points=5,
        shooting_max_iter=3,
        device=device
    )
    
    result_parallel = parallel_model(c_source_batch, c_target_batch, wl_batch)
    print(f"Parallel output shape: {result_parallel['absorbance'].shape}")
    
    # Benchmark
    print("\nBenchmarking...")
    import time
    
    models = [
        ("Standard", model, 256),
        ("Parallel", parallel_model, 1024)
    ]
    
    for name, test_model, test_batch in models:
        test_model.eval()
        
        # Create test data
        c_s = torch.randn(test_batch, device=device) * 0.5
        c_t = torch.randn(test_batch, device=device) * 0.5
        wl = torch.randn(test_batch, device=device)
        
        # Warmup
        with torch.no_grad():
            _ = test_model(c_s[:10], c_t[:10], wl[:10])
        
        # Benchmark
        start = time.time()
        with torch.no_grad():
            output = test_model(c_s, c_t, wl)
        
        if device.type == 'mps':
            torch.mps.synchronize()
        
        elapsed = time.time() - start
        throughput = test_batch / elapsed
        
        print(f"{name:10s}: {elapsed:.3f}s for {test_batch} samples ({throughput:.0f} samples/s)")
    
    print("\nGeodesic model tests passed!")


if __name__ == "__main__":
    test_geodesic_model()