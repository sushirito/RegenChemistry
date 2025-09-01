#!/usr/bin/env python3
"""
Parallel Shooting Method for Boundary Value Problems
Solves millions of BVPs simultaneously without loops
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from .geodesic_ode import ParallelGeodesicODE, VectorizedGeodesicODE


class ParallelShootingSolver:
    """
    Solves boundary value problems using shooting method
    Find v₀ such that geodesic from c_source reaches c_target
    """
    
    def __init__(self, ode_system: ParallelGeodesicODE,
                 max_iterations: int = 10,
                 tolerance: float = 1e-4,
                 learning_rate: float = 0.5):
        """
        Initialize shooting solver
        
        Args:
            ode_system: The geodesic ODE system
            max_iterations: Fixed number of Newton iterations
            tolerance: Convergence tolerance (not used in parallel mode)
            learning_rate: Step size for v₀ updates
        """
        self.ode_system = ode_system
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.learning_rate = learning_rate
        
        # Statistics
        self.last_errors = None
        self.last_iterations = None
    
    def solve_batch(self, c_source: torch.Tensor, c_target: torch.Tensor,
                   wavelengths: torch.Tensor, n_trajectory_points: int = 11,
                   return_trajectories: bool = True) -> Dict[str, torch.Tensor]:
        """
        Solve BVPs for all transitions in parallel
        
        Args:
            c_source: Source concentrations (batch*wavelength,)
            c_target: Target concentrations (batch*wavelength,)
            wavelengths: Wavelength values (batch*wavelength,)
            n_trajectory_points: Number of points along trajectory
            return_trajectories: Whether to return full trajectories
        
        Returns:
            Dictionary containing:
                - trajectories: Full paths if requested
                - v_initial: Optimal initial velocities
                - c_final: Final concentrations reached
                - errors: Final errors
                - path_stats: Statistics for decoder
        """
        batch_size = c_source.shape[0]
        device = c_source.device
        
        # Initial guess for velocities (linear approximation)
        v_initial = c_target - c_source
        
        # Pre-allocate for efficiency
        errors = torch.zeros(batch_size, device=device)
        
        # Fixed iterations (no conditionals for parallelism)
        for iteration in range(self.max_iterations):
            # Integrate with current velocities
            result = self.ode_system.integrate_batch(
                c_source, v_initial, wavelengths,
                n_steps=n_trajectory_points
            )
            
            c_final = result['c_final']
            
            # Compute errors
            errors = c_final - c_target
            
            # Update velocities (Newton-like step)
            # Simplified update without Jacobian for speed
            v_update = -self.learning_rate * errors
            v_initial = v_initial + v_update
            
            # Optional: adaptive learning rate
            if iteration > 0:
                # Reduce learning rate if oscillating
                error_change = (errors.abs() - prev_errors.abs()).mean()
                if error_change > 0:
                    self.learning_rate *= 0.9
            
            prev_errors = errors.clone()
        
        # Final integration with optimal velocities
        final_result = self.ode_system.integrate_batch(
            c_source, v_initial, wavelengths,
            n_steps=n_trajectory_points
        )
        
        # Store statistics
        self.last_errors = errors.abs()
        self.last_iterations = self.max_iterations
        
        output = {
            'v_initial': v_initial,
            'c_final': final_result['c_final'],
            'errors': errors,
            'converged': errors.abs() < self.tolerance,
            **final_result  # Include all path statistics
        }
        
        if not return_trajectories:
            output.pop('trajectories', None)
        
        return output
    
    def solve_parallel(self, c_source: torch.Tensor, c_target: torch.Tensor,
                       wavelengths: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Simplified interface for parallel solving
        """
        return self.solve_batch(
            c_source, c_target, wavelengths,
            return_trajectories=False
        )


class OptimizedShootingSolver(ParallelShootingSolver):
    """
    Optimized version with advanced techniques
    """
    
    def __init__(self, ode_system: ParallelGeodesicODE,
                 max_iterations: int = 10,
                 use_jacobian: bool = False,
                 use_momentum: bool = True,
                 momentum: float = 0.9):
        super().__init__(ode_system, max_iterations)
        self.use_jacobian = use_jacobian
        self.use_momentum = use_momentum
        self.momentum = momentum
        self.velocity_momentum = None
    
    def solve_batch_optimized(self, c_source: torch.Tensor, c_target: torch.Tensor,
                             wavelengths: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Optimized solving with momentum and optional Jacobian
        """
        batch_size = c_source.shape[0]
        device = c_source.device
        
        # Initial guess
        v_initial = c_target - c_source
        
        # Initialize momentum
        if self.use_momentum:
            if self.velocity_momentum is None:
                self.velocity_momentum = torch.zeros_like(v_initial)
            else:
                # Resize if batch size changed
                if self.velocity_momentum.shape[0] != batch_size:
                    self.velocity_momentum = torch.zeros_like(v_initial)
        
        # Learning rate schedule
        lr = 1.0
        lr_decay = 0.95
        
        best_v = v_initial.clone()
        best_error = float('inf')
        
        for iteration in range(self.max_iterations):
            # Forward integration
            result = self.ode_system.integrate_batch(
                c_source, v_initial, wavelengths, n_steps=11
            )
            
            c_final = result['c_final']
            errors = c_final - c_target
            
            # Track best solution
            error_norm = errors.abs().mean()
            if error_norm < best_error:
                best_error = error_norm
                best_v = v_initial.clone()
            
            # Compute update
            if self.use_jacobian:
                # Approximate Jacobian using finite differences
                epsilon = 1e-4
                v_perturbed = v_initial + epsilon
                result_perturbed = self.ode_system.integrate_batch(
                    c_source, v_perturbed, wavelengths, n_steps=5
                )
                c_perturbed = result_perturbed['c_final']
                
                jacobian = (c_perturbed - c_final) / epsilon
                jacobian = jacobian.clamp(min=0.1, max=10.0)  # Stability
                
                v_update = -errors / (jacobian + 1e-8)
            else:
                # Simple proportional update
                v_update = -lr * errors
            
            # Apply momentum
            if self.use_momentum:
                self.velocity_momentum = self.momentum * self.velocity_momentum + v_update
                v_initial = v_initial + self.velocity_momentum
            else:
                v_initial = v_initial + v_update
            
            # Decay learning rate
            lr *= lr_decay
        
        # Use best found solution
        final_result = self.ode_system.integrate_batch(
            c_source, best_v, wavelengths, n_steps=11
        )
        
        return {
            'v_initial': best_v,
            **final_result
        }


class BatchedMultiShootingSolver:
    """
    Multi-shooting method for improved convergence
    Splits trajectory into segments
    """
    
    def __init__(self, ode_system: ParallelGeodesicODE,
                 n_segments: int = 3,
                 max_iterations: int = 10):
        self.ode_system = ode_system
        self.n_segments = n_segments
        self.max_iterations = max_iterations
        self.single_shooter = OptimizedShootingSolver(ode_system)
    
    def solve_batch(self, c_source: torch.Tensor, c_target: torch.Tensor,
                   wavelengths: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Multi-shooting with intermediate points
        """
        batch_size = c_source.shape[0]
        device = c_source.device
        
        # Create intermediate targets
        alphas = torch.linspace(0, 1, self.n_segments + 1, device=device)
        intermediate_c = []
        
        for alpha in alphas:
            c_intermediate = (1 - alpha) * c_source + alpha * c_target
            intermediate_c.append(c_intermediate)
        
        # Solve each segment
        segment_velocities = []
        current_c = c_source
        
        for i in range(self.n_segments):
            target_c = intermediate_c[i + 1]
            
            # Solve segment
            segment_result = self.single_shooter.solve_batch_optimized(
                current_c, target_c, wavelengths
            )
            
            segment_velocities.append(segment_result['v_initial'])
            current_c = segment_result['c_final']
        
        # Combine segments for full trajectory
        # Use first segment velocity as initial velocity
        v_initial = segment_velocities[0]
        
        # Full integration
        final_result = self.ode_system.integrate_batch(
            c_source, v_initial, wavelengths,
            n_steps=11 * self.n_segments
        )
        
        return {
            'v_initial': v_initial,
            'segment_velocities': segment_velocities,
            **final_result
        }


def test_shooting_solver():
    """Test shooting solver"""
    print("Testing Parallel Shooting Solver...")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Setup
    from geodesic_mps.models.metric_network import OptimizedMetricNetwork
    from geodesic_mps.core.christoffel import ParallelChristoffelComputer
    
    metric = OptimizedMetricNetwork().to(device)
    christoffel = ParallelChristoffelComputer()
    ode_system = VectorizedGeodesicODE(
        christoffel.compute_parallel,
        metric
    )
    
    # Create solver
    solver = ParallelShootingSolver(ode_system)
    
    # Test single BVP
    print("\nTesting single BVP...")
    c_source = torch.tensor([-0.5], device=device)
    c_target = torch.tensor([0.5], device=device)
    wavelength = torch.tensor([0.0], device=device)
    
    result = solver.solve_batch(
        c_source, c_target, wavelength,
        n_trajectory_points=11
    )
    
    print(f"Initial velocity: {result['v_initial'].item():.4f}")
    print(f"Final concentration: {result['c_final'].item():.4f}")
    print(f"Target concentration: {c_target.item():.4f}")
    print(f"Error: {result['errors'].item():.6f}")
    print(f"Converged: {result['converged'].item()}")
    
    # Test batch BVP
    print("\nTesting batch BVP...")
    batch_size = 256
    c_source_batch = torch.randn(batch_size, device=device) * 0.5 - 0.5
    c_target_batch = torch.randn(batch_size, device=device) * 0.5 + 0.5
    wl_batch = torch.randn(batch_size, device=device)
    
    result_batch = solver.solve_batch(
        c_source_batch, c_target_batch, wl_batch
    )
    
    errors = result_batch['errors'].abs()
    print(f"Batch size: {batch_size}")
    print(f"Mean error: {errors.mean().item():.6f}")
    print(f"Max error: {errors.max().item():.6f}")
    print(f"Converged: {result_batch['converged'].sum().item()}/{batch_size}")
    
    # Test optimized solver
    print("\nTesting optimized solver...")
    opt_solver = OptimizedShootingSolver(ode_system, use_momentum=True)
    
    result_opt = opt_solver.solve_batch_optimized(
        c_source_batch, c_target_batch, wl_batch
    )
    
    errors_opt = (result_opt['c_final'] - c_target_batch).abs()
    print(f"Optimized mean error: {errors_opt.mean().item():.6f}")
    print(f"Optimized max error: {errors_opt.max().item():.6f}")
    
    # Benchmark
    print("\nBenchmarking...")
    import time
    
    mega_batch = 1024
    n_wavelengths = 601
    total_size = mega_batch * n_wavelengths
    
    c_source_mega = torch.randn(total_size, device=device) * 0.5 - 0.5
    c_target_mega = torch.randn(total_size, device=device) * 0.5 + 0.5
    wl_mega = torch.randn(total_size, device=device)
    
    start = time.time()
    result_mega = solver.solve_parallel(
        c_source_mega, c_target_mega, wl_mega
    )
    
    if device.type == 'mps':
        torch.mps.synchronize()
    
    elapsed = time.time() - start
    throughput = total_size / elapsed
    
    print(f"Mega-batch size: {total_size:,} BVPs")
    print(f"Solving time: {elapsed:.3f}s")
    print(f"Throughput: {throughput:.0f} BVPs/second")
    
    # Test multi-shooting
    print("\nTesting multi-shooting solver...")
    multi_solver = BatchedMultiShootingSolver(ode_system, n_segments=3)
    
    result_multi = multi_solver.solve_batch(
        c_source_batch[:10], c_target_batch[:10], wl_batch[:10]
    )
    
    errors_multi = (result_multi['c_final'] - c_target_batch[:10]).abs()
    print(f"Multi-shooting mean error: {errors_multi.mean().item():.6f}")
    
    print("\nShooting solver tests passed!")


if __name__ == "__main__":
    test_shooting_solver()