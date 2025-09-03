"""
Parallel shooting method for boundary value problems
Solves for initial velocities that connect source to target concentrations
Optimized for M1 Mac MPS acceleration
"""

import torch
from typing import Dict, Optional


class ShootingSolver:
    """Solves BVP using parallel shooting method for massive batches"""
    
    def __init__(self,
                 geodesic_integrator: 'GeodesicIntegrator',
                 max_iterations: int = 10,
                 tolerance: float = 1e-4,
                 learning_rate: float = 0.5,
                 device: torch.device = torch.device('mps')):
        """
        Initialize shooting solver
        
        Args:
            geodesic_integrator: Integrator for geodesic ODEs
            max_iterations: Maximum shooting iterations
            tolerance: Convergence tolerance for endpoint error
            learning_rate: Step size for velocity updates
            device: Computation device (MPS for M1 Mac)
        """
        self.integrator = geodesic_integrator
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.learning_rate = learning_rate
        self.device = device
        
    def solve_batch(self,
                   c_sources: torch.Tensor,
                   c_targets: torch.Tensor,
                   wavelengths: torch.Tensor,
                   n_trajectory_points: int = 50) -> Dict[str, torch.Tensor]:
        """
        Solve BVP for batch of geodesics
        Find initial velocities v₀ such that geodesic(c_source, v₀, t=1) = c_target
        
        Args:
            c_sources: Source concentrations [batch_size]
            c_targets: Target concentrations [batch_size]
            wavelengths: Wavelengths [batch_size]
            n_trajectory_points: Points for trajectory discretization
            
        Returns:
            Dictionary with:
                - initial_velocities: Optimal v₀ [batch_size]
                - trajectories: Full geodesic paths [n_points, batch_size, 3]
                - final_absorbance: Predicted absorbance values [batch_size]
                - final_errors: Endpoint errors [batch_size]
                - convergence_mask: Which geodesics converged [batch_size]
        """
        batch_size = c_sources.shape[0]
        
        # Initial guess: linear interpolation velocity
        v_initial = c_targets - c_sources
        
        # Track best solutions
        best_velocities = v_initial.clone()
        best_errors = torch.full((batch_size,), float('inf'), device=self.device)
        convergence_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        # Fixed iterations for parallelism (no early stopping)
        for iteration in range(self.max_iterations):
            # Integrate with current velocities (A starts at 0)
            initial_A = torch.zeros_like(c_sources)
            initial_states = torch.stack([c_sources, v_initial, initial_A], dim=1)
            t_span = torch.tensor([0.0, 1.0], device=self.device, dtype=torch.float32)
            
            # Quick integration for shooting (only need endpoints)
            results = self.integrator.integrate_batch(
                initial_states=initial_states,
                wavelengths=wavelengths,
                t_span=t_span,
                method='rk4',  # Faster for shooting
                rtol=1e-3,  # Looser tolerance for shooting
                atol=1e-5
            )
            
            # Get final concentrations
            c_final = results['final_states'][:, 0]
            
            # Compute errors
            errors = (c_final - c_targets).abs()
            
            # Update best solutions
            improved_mask = errors < best_errors
            best_velocities[improved_mask] = v_initial[improved_mask]
            best_errors[improved_mask] = errors[improved_mask]
            
            # Check convergence
            newly_converged = (errors < self.tolerance) & ~convergence_mask
            convergence_mask |= newly_converged
            
            # Compute velocity update using gradient
            # Simple gradient: error proportional to velocity correction needed
            velocity_gradient = (c_final - c_targets)
            
            # Update velocities (except for converged ones)
            update_mask = ~convergence_mask
            v_initial[update_mask] -= self.learning_rate * velocity_gradient[update_mask]
            
            # Adaptive learning rate decay
            if iteration > 0 and iteration % 3 == 0:
                self.learning_rate *= 0.8
                
        # Final integration with best velocities for full trajectories
        initial_A = torch.zeros_like(c_sources)
        initial_states = torch.stack([c_sources, best_velocities, initial_A], dim=1)
        t_span = torch.linspace(0, 1, n_trajectory_points, device=self.device, dtype=torch.float32)
        
        final_results = self.integrator.integrate_batch(
            initial_states=initial_states,
            wavelengths=wavelengths,
            t_span=t_span,
            rtol=1e-5,
            atol=1e-7
        )
        
        # Prepare output
        return {
            'initial_velocities': best_velocities,
            'trajectories': final_results['trajectories'],
            'final_absorbance': final_results['final_absorbance'],
            'final_errors': best_errors,
            'convergence_mask': convergence_mask,
            'convergence_rate': convergence_mask.float().mean(),
            'mean_error': best_errors.mean()
        }
        
    def solve_parallel(self,
                       concentration_pairs: torch.Tensor,
                       wavelength_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Solve all 18,030 geodesics in parallel for complete dataset
        
        Args:
            concentration_pairs: All concentration transitions [30, 2] 
            wavelength_grid: All wavelengths [601]
            
        Returns:
            Complete results for all geodesics
        """
        n_pairs, _ = concentration_pairs.shape
        n_wavelengths = wavelength_grid.shape[0]
        total_geodesics = n_pairs * n_wavelengths
        
        print(f"Solving {total_geodesics} geodesics in parallel on M1 Mac...")
        
        # Expand to all combinations
        c_sources_expanded = concentration_pairs[:, 0].repeat_interleave(n_wavelengths)
        c_targets_expanded = concentration_pairs[:, 1].repeat_interleave(n_wavelengths)
        wavelengths_expanded = wavelength_grid.repeat(n_pairs)
        
        # Move to device
        c_sources_expanded = c_sources_expanded.to(self.device)
        c_targets_expanded = c_targets_expanded.to(self.device)
        wavelengths_expanded = wavelengths_expanded.to(self.device)
        
        # Solve all geodesics
        results = self.solve_batch(
            c_sources=c_sources_expanded,
            c_targets=c_targets_expanded,
            wavelengths=wavelengths_expanded
        )
        
        # Reshape results to [n_pairs, n_wavelengths, ...]
        def reshape_result(tensor):
            if tensor.dim() == 1:
                return tensor.view(n_pairs, n_wavelengths)
            elif tensor.dim() == 2:
                return tensor.view(n_pairs, n_wavelengths, -1)
            elif tensor.dim() == 3:
                return tensor.view(-1, n_pairs, n_wavelengths, tensor.shape[-1])
            else:
                return tensor
                
        reshaped_results = {}
        for key, value in results.items():
            reshaped_results[key] = reshape_result(value)
            
        return reshaped_results