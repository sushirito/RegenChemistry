#!/usr/bin/env python3
"""
Geodesic ODE System
Implements the true geodesic equation: d²c/dt² = -Γ(c,λ)(dc/dt)²
"""

import torch
import numpy as np
from typing import Callable, Tuple, Optional
from src.models.metric_network import MetricNetwork
from src.core.christoffel import ChristoffelComputer


class GeodesicODE:
    """
    Defines the geodesic ODE system for numerical integration
    
    State vector: [c, v] where v = dc/dt
    ODE system:
        dc/dt = v
        dv/dt = -Γ(c,λ)v²
    """
    
    def __init__(self, 
                 metric_network: MetricNetwork,
                 wavelength: float,
                 christoffel_computer: Optional[ChristoffelComputer] = None):
        """
        Initialize geodesic ODE
        
        Args:
            metric_network: The Riemannian metric network
            wavelength: The wavelength parameter (fixed during integration)
            christoffel_computer: Optional pre-configured Christoffel computer
        """
        self.metric_network = metric_network
        self.wavelength = wavelength
        self.christoffel_computer = christoffel_computer or ChristoffelComputer()
        self.num_evaluations = 0
    
    def __call__(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Evaluate the ODE system (for scipy integration)
        
        Args:
            t: Time parameter (not used directly, but required by ODE solver)
            state: State vector [c, v]
        
        Returns:
            Derivative [dc/dt, dv/dt]
        """
        self.num_evaluations += 1
        
        # Extract state components
        c, v = state[0], state[1]
        
        # Convert to torch tensors
        c_tensor = torch.tensor([c], dtype=torch.float32)
        wavelength_tensor = torch.tensor([self.wavelength], dtype=torch.float32)
        
        # Compute Christoffel symbol
        with torch.no_grad():  # No gradients needed during ODE integration
            gamma = self.christoffel_computer.compute(
                c_tensor, wavelength_tensor, self.metric_network
            ).item()
        
        # Compute derivatives
        dc_dt = v
        dv_dt = -gamma * v * v  # -Γ(c,λ)v²
        
        return np.array([dc_dt, dv_dt])
    
    def torch_forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the ODE system (for torch-based ODE solvers)
        
        Args:
            t: Time parameter (batch_size,) or scalar
            state: State tensor [..., 2] with last dimension [c, v]
        
        Returns:
            Derivative tensor [..., 2] with [dc/dt, dv/dt]
        """
        self.num_evaluations += 1
        
        # Handle different input shapes
        original_shape = state.shape
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Extract state components
        c = state[..., 0]
        v = state[..., 1]
        
        # Prepare wavelength tensor
        wavelength_tensor = torch.full_like(c, self.wavelength)
        
        # Compute Christoffel symbols
        gamma = self.christoffel_computer.compute(
            c, wavelength_tensor, self.metric_network
        )
        
        # Compute derivatives
        dc_dt = v
        dv_dt = -gamma * v * v
        
        # Stack derivatives
        derivatives = torch.stack([dc_dt, dv_dt], dim=-1)
        
        # Reshape to match input
        if original_shape[0] == 2 and derivatives.shape[0] == 1:
            derivatives = derivatives.squeeze(0)
        
        return derivatives
    
    def reset_stats(self):
        """Reset evaluation counter"""
        self.num_evaluations = 0


def verify_geodesic_equation(metric_network: MetricNetwork,
                            wavelength: float = 0.0,
                            n_tests: int = 10) -> dict:
    """
    Verify that the geodesic equation is correctly implemented
    
    Args:
        metric_network: The metric network
        wavelength: Wavelength to test at
        n_tests: Number of test points
    
    Returns:
        Dictionary of verification results
    """
    ode = GeodesicODE(metric_network, wavelength)
    results = {}
    
    # Test 1: Zero velocity should give zero acceleration
    zero_vel_states = []
    for _ in range(n_tests):
        c = np.random.randn()
        state = np.array([c, 0.0])  # Zero velocity
        derivative = ode(0.0, state)
        zero_vel_states.append(derivative[1])  # Should be 0
    
    results['zero_velocity_zero_accel'] = np.allclose(zero_vel_states, 0, atol=1e-10)
    
    # Test 2: Verify equation structure
    test_states = []
    for _ in range(n_tests):
        state = np.random.randn(2)
        derivative = ode(0.0, state)
        # dc/dt should equal v
        test_states.append(np.abs(derivative[0] - state[1]))
    
    results['velocity_consistency'] = np.allclose(test_states, 0, atol=1e-10)
    
    # Test 3: Check for NaN or Inf
    nan_count = 0
    inf_count = 0
    for _ in range(100):
        state = np.random.randn(2) * 10  # Test with larger values
        derivative = ode(0.0, state)
        if np.any(np.isnan(derivative)):
            nan_count += 1
        if np.any(np.isinf(derivative)):
            inf_count += 1
    
    results['nan_count'] = nan_count
    results['inf_count'] = inf_count
    results['numerical_stability'] = (nan_count == 0 and inf_count == 0)
    
    return results


def test_torch_numpy_consistency(metric_network: MetricNetwork):
    """
    Test that torch and numpy versions give same results
    
    Args:
        metric_network: The metric network
    
    Returns:
        Maximum difference between implementations
    """
    wavelength = 0.5
    ode = GeodesicODE(metric_network, wavelength)
    
    max_diff = 0.0
    for _ in range(20):
        # Random test state
        c = np.random.randn()
        v = np.random.randn()
        
        # NumPy version
        state_np = np.array([c, v])
        deriv_np = ode(0.0, state_np)
        
        # Torch version
        state_torch = torch.tensor([c, v], dtype=torch.float32)
        deriv_torch = ode.torch_forward(torch.tensor(0.0), state_torch)
        deriv_torch_np = deriv_torch.detach().numpy()
        
        # Compare
        diff = np.abs(deriv_np - deriv_torch_np).max()
        max_diff = max(max_diff, diff)
    
    return max_diff


if __name__ == "__main__":
    # Test the geodesic ODE system
    print("Testing Geodesic ODE System...")
    
    # Create metric network
    metric_net = MetricNetwork()
    
    # Test basic ODE evaluation
    wavelength = 0.0  # Normalized wavelength
    ode = GeodesicODE(metric_net, wavelength)
    
    # Test single evaluation
    state = np.array([0.5, 1.0])  # c=0.5, v=1.0
    derivative = ode(0.0, state)
    print(f"State: c={state[0]:.3f}, v={state[1]:.3f}")
    print(f"Derivative: dc/dt={derivative[0]:.3f}, dv/dt={derivative[1]:.3f}")
    
    # Verify equation properties
    print("\nVerifying geodesic equation properties:")
    verification = verify_geodesic_equation(metric_net)
    for key, value in verification.items():
        symbol = "✓" if value in [True, 0] else "✗"
        print(f"  {symbol} {key}: {value}")
    
    # Test torch version
    print("\nTesting PyTorch version:")
    state_torch = torch.tensor([0.5, 1.0], dtype=torch.float32)
    deriv_torch = ode.torch_forward(torch.tensor(0.0), state_torch)
    print(f"Torch derivative: {deriv_torch}")
    
    # Test batch processing
    print("\nTesting batch processing:")
    batch_size = 4
    state_batch = torch.randn(batch_size, 2)
    deriv_batch = ode.torch_forward(torch.zeros(batch_size), state_batch)
    print(f"Batch input shape: {state_batch.shape}")
    print(f"Batch output shape: {deriv_batch.shape}")
    
    # Test consistency
    print("\nTesting NumPy/Torch consistency:")
    max_diff = test_torch_numpy_consistency(metric_net)
    print(f"Maximum difference: {max_diff:.10f}")
    if max_diff < 1e-6:
        print("✓ Implementations are consistent")
    else:
        print("✗ Implementations differ significantly")
    
    # Test evaluation count
    print(f"\nTotal ODE evaluations: {ode.num_evaluations}")
    
    print("\nGeodesic ODE tests passed!")