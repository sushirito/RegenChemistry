#!/usr/bin/env python3
"""
Shooting Method Solver for Geodesic Boundary Value Problem
Finds initial velocity v₀ such that geodesic from c_source reaches c_target
"""

import torch
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar, minimize
from typing import Tuple, Optional, Dict
from metric_network import MetricNetwork
from geodesic_ode import GeodesicODE


class ShootingSolver:
    """
    Solves the boundary value problem for geodesics using shooting method
    
    Problem: Find v₀ such that integrating from [c_source, v₀] at t=0 
             reaches c_target at t=1
    """
    
    def __init__(self, 
                 metric_network: MetricNetwork,
                 max_iterations: int = 50,
                 tolerance: float = 1e-4,
                 integration_method: str = 'RK45'):
        """
        Initialize shooting solver
        
        Args:
            metric_network: The Riemannian metric network
            max_iterations: Maximum optimization iterations
            tolerance: Tolerance for reaching target
            integration_method: ODE integration method
        """
        self.metric_network = metric_network
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.integration_method = integration_method
        self.solve_stats = {
            'total_solves': 0,
            'successful_solves': 0,
            'failed_solves': 0,
            'average_iterations': 0
        }
    
    def objective_function(self, 
                          v0: float, 
                          c_source: float, 
                          c_target: float,
                          wavelength: float) -> float:
        """
        Objective function for shooting method
        Returns squared error between final position and target
        
        Args:
            v0: Initial velocity to test
            c_source: Starting concentration
            c_target: Target concentration
            wavelength: Wavelength parameter
        
        Returns:
            Squared error (c_final - c_target)²
        """
        # Create ODE system
        ode = GeodesicODE(self.metric_network, wavelength)
        
        # Initial state
        initial_state = np.array([c_source, v0])
        
        # Integrate ODE from t=0 to t=1
        try:
            solution = solve_ivp(
                ode,
                t_span=[0.0, 1.0],
                y0=initial_state,
                method=self.integration_method,
                dense_output=False,
                rtol=1e-6,
                atol=1e-8
            )
            
            if not solution.success:
                return 1e10  # Large penalty for integration failure
            
            # Get final concentration
            c_final = solution.y[0, -1]
            
            # Return squared error
            return (c_final - c_target) ** 2
            
        except Exception as e:
            # Return large penalty for any numerical issues
            return 1e10
    
    def solve(self, 
              c_source: float,
              c_target: float, 
              wavelength: float,
              return_trajectory: bool = True,
              n_trajectory_points: int = 11) -> Dict:
        """
        Solve the boundary value problem
        
        Args:
            c_source: Starting concentration (normalized)
            c_target: Target concentration (normalized)
            wavelength: Wavelength parameter (normalized)
            return_trajectory: Whether to return full geodesic path
            n_trajectory_points: Number of points along trajectory
        
        Returns:
            Dictionary containing:
                - success: Whether solution was found
                - v0: Initial velocity
                - trajectory: Full path (if requested)
                - error: Final position error
                - iterations: Number of iterations used
        """
        self.solve_stats['total_solves'] += 1
        
        # Initial guess: linear velocity
        v0_initial = c_target - c_source
        
        # Define objective for this specific BVP
        def obj(v0):
            return self.objective_function(v0, c_source, c_target, wavelength)
        
        # Try different optimization methods
        result = None
        success = False
        
        # Method 1: Bounded scalar minimization (fastest for 1D)
        try:
            # Set reasonable bounds on velocity
            v_max = abs(c_target - c_source) * 5
            result = minimize_scalar(
                obj,
                bounds=(-v_max, v_max),
                method='bounded',
                options={'maxiter': self.max_iterations}
            )
            v0_optimal = result.x
            final_error = np.sqrt(result.fun)
            success = final_error < self.tolerance
            iterations = result.nfev
        except:
            pass
        
        # Method 2: General minimization as fallback
        if not success:
            try:
                result = minimize(
                    obj,
                    x0=[v0_initial],
                    method='L-BFGS-B',
                    bounds=[(-10, 10)],
                    options={'maxiter': self.max_iterations}
                )
                v0_optimal = result.x[0]
                final_error = np.sqrt(result.fun)
                success = final_error < self.tolerance
                iterations = result.nit
            except:
                v0_optimal = v0_initial
                final_error = float('inf')
                success = False
                iterations = 0
        
        # Update statistics
        if success:
            self.solve_stats['successful_solves'] += 1
        else:
            self.solve_stats['failed_solves'] += 1
        
        # Build result dictionary
        result_dict = {
            'success': success,
            'v0': v0_optimal,
            'error': final_error,
            'iterations': iterations
        }
        
        # Get full trajectory if requested
        if return_trajectory:
            trajectory = self.integrate_geodesic(
                c_source, v0_optimal, wavelength, n_trajectory_points
            )
            result_dict['trajectory'] = trajectory
        
        return result_dict
    
    def integrate_geodesic(self,
                          c_source: float,
                          v0: float,
                          wavelength: float,
                          n_points: int = 11) -> np.ndarray:
        """
        Integrate geodesic with given initial conditions
        
        Args:
            c_source: Starting concentration
            v0: Initial velocity
            wavelength: Wavelength parameter
            n_points: Number of points to evaluate
        
        Returns:
            Trajectory array of shape (n_points, 2) with [c, v] at each point
        """
        # Create ODE system
        ode = GeodesicODE(self.metric_network, wavelength)
        
        # Initial state
        initial_state = np.array([c_source, v0])
        
        # Time points to evaluate
        t_eval = np.linspace(0, 1, n_points)
        
        # Integrate ODE
        try:
            solution = solve_ivp(
                ode,
                t_span=[0.0, 1.0],
                y0=initial_state,
                t_eval=t_eval,
                method=self.integration_method,
                rtol=1e-6,
                atol=1e-8
            )
            
            if solution.success:
                # Return trajectory: (n_points, 2)
                return solution.y.T
            else:
                # Return linear interpolation as fallback
                c_linear = np.linspace(c_source, c_source + v0, n_points)
                v_linear = np.full(n_points, v0)
                return np.stack([c_linear, v_linear], axis=1)
                
        except:
            # Return linear interpolation as fallback
            c_linear = np.linspace(c_source, c_source + v0, n_points)
            v_linear = np.full(n_points, v0)
            return np.stack([c_linear, v_linear], axis=1)
    
    def solve_batch(self,
                   c_sources: torch.Tensor,
                   c_targets: torch.Tensor,
                   wavelengths: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Solve multiple BVPs in batch
        
        Args:
            c_sources: Source concentrations (batch_size,)
            c_targets: Target concentrations (batch_size,)
            wavelengths: Wavelengths (batch_size,)
        
        Returns:
            Dictionary with batch results
        """
        batch_size = c_sources.shape[0]
        v0_list = []
        trajectory_list = []
        success_list = []
        
        for i in range(batch_size):
            result = self.solve(
                c_sources[i].item(),
                c_targets[i].item(),
                wavelengths[i].item(),
                return_trajectory=True
            )
            
            v0_list.append(result['v0'])
            trajectory_list.append(torch.tensor(result['trajectory'], dtype=torch.float32))
            success_list.append(result['success'])
        
        return {
            'v0': torch.tensor(v0_list, dtype=torch.float32),
            'trajectories': torch.stack(trajectory_list),
            'success': torch.tensor(success_list, dtype=torch.bool)
        }
    
    def get_stats(self) -> Dict:
        """Get solver statistics"""
        return self.solve_stats.copy()


def test_shooting_solver():
    """Test the shooting solver with various scenarios"""
    print("Testing Shooting Solver...")
    
    # Create metric network and solver
    metric_net = MetricNetwork()
    solver = ShootingSolver(metric_net)
    
    # Test 1: Same source and target (should have v0 ≈ 0)
    print("\nTest 1: Same source and target")
    result = solver.solve(0.5, 0.5, 0.0)
    print(f"  v0 = {result['v0']:.6f} (expected ≈ 0)")
    print(f"  Error = {result['error']:.8f}")
    print(f"  Success = {result['success']}")
    
    # Test 2: Different concentrations
    print("\nTest 2: Different concentrations")
    result = solver.solve(-0.5, 0.5, 0.0)
    print(f"  c_source = -0.5, c_target = 0.5")
    print(f"  v0 = {result['v0']:.6f}")
    print(f"  Error = {result['error']:.8f}")
    print(f"  Success = {result['success']}")
    
    if result['success']:
        trajectory = result['trajectory']
        print(f"  Final c = {trajectory[-1, 0]:.6f} (target = 0.5)")
    
    # Test 3: Multiple wavelengths
    print("\nTest 3: Multiple wavelengths")
    wavelengths = [-0.5, 0.0, 0.5]
    for wl in wavelengths:
        result = solver.solve(0.0, 0.8, wl)
        print(f"  λ = {wl:+.1f}: v0 = {result['v0']:.4f}, success = {result['success']}")
    
    # Test 4: Batch solving
    print("\nTest 4: Batch solving")
    batch_size = 4
    c_sources = torch.tensor([-0.5, -0.3, 0.0, 0.2])
    c_targets = torch.tensor([0.5, 0.3, 0.5, 0.8])
    wavelengths = torch.zeros(batch_size)
    
    batch_results = solver.solve_batch(c_sources, c_targets, wavelengths)
    print(f"  Batch size: {batch_size}")
    print(f"  Initial velocities: {batch_results['v0']}")
    print(f"  Success rate: {batch_results['success'].float().mean().item():.1%}")
    
    # Test 5: Trajectory properties
    print("\nTest 5: Trajectory properties")
    result = solver.solve(-0.8, 0.8, 0.0, n_trajectory_points=21)
    if result['success']:
        trajectory = result['trajectory']
        print(f"  Trajectory shape: {trajectory.shape}")
        print(f"  Start: c = {trajectory[0, 0]:.4f}, v = {trajectory[0, 1]:.4f}")
        print(f"  End:   c = {trajectory[-1, 0]:.4f}, v = {trajectory[-1, 1]:.4f}")
        
        # Check smoothness
        c_path = trajectory[:, 0]
        c_diff = np.diff(c_path)
        print(f"  Path monotonic: {np.all(c_diff >= 0) or np.all(c_diff <= 0)}")
    
    # Print statistics
    print("\nSolver Statistics:")
    stats = solver.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nShooting Solver tests completed!")


if __name__ == "__main__":
    test_shooting_solver()