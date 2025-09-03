"""
Massive batch geodesic ODE integration
Optimized for M1 Mac MPS processing of 18,030 simultaneous geodesics
"""

import torch
from torchdiffeq import odeint, odeint_adjoint
from typing import Dict, Optional, Callable


class GeodesicIntegrator:
    """Integrates coupled geodesic-spectral ODEs for massive batches simultaneously"""
    
    def __init__(self,
                 christoffel_computer: 'ChristoffelComputer',
                 spectral_flow_network: torch.nn.Module,
                 device: torch.device = torch.device('mps'),
                 use_adjoint: bool = True):
        """
        Initialize geodesic integrator with coupled dynamics
        
        Args:
            christoffel_computer: Pre-computed Christoffel symbols
            spectral_flow_network: Network for spectral flow dA/dt = f(c,v,λ)
            device: Computation device (MPS for M1 Mac)
            use_adjoint: Use adjoint method for memory efficiency
        """
        self.christoffel_computer = christoffel_computer
        self.spectral_flow_network = spectral_flow_network
        self.device = device
        self.use_adjoint = use_adjoint
        
    def integrate_batch(self,
                       initial_states: torch.Tensor,
                       wavelengths: torch.Tensor,
                       t_span: torch.Tensor,
                       method: str = 'dopri5',
                       rtol: float = 1e-5,
                       atol: float = 1e-7) -> Dict[str, torch.Tensor]:
        """
        Integrate geodesic ODEs for massive batch
        
        State vector: [c, v, A] where:
            c: concentration
            v: velocity dc/dt
            A: absorbance (evolves through coupled ODE)
        
        Args:
            initial_states: Initial [c, v, A] states [batch_size, 3]
            wavelengths: Wavelength values [batch_size]
            t_span: Time points for integration [n_time_points]
            method: ODE solver method
            rtol: Relative tolerance
            atol: Absolute tolerance
            
        Returns:
            Dictionary with:
                - trajectories: Full state evolution [n_time_points, batch_size, 3]
                - final_states: Final states [batch_size, 3]
                - final_absorbance: Final A values [batch_size]
        """
        batch_size = initial_states.shape[0]
        assert initial_states.shape[1] == 3, "State must be [c, v, A] with dimension 3"
        
        # Store wavelengths for ODE function
        self._current_wavelengths = wavelengths
        
        # Create coupled ODE function
        def coupled_geodesic_ode(t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
            """
            Coupled geodesic-spectral ODE system:
            dc/dt = v
            dv/dt = -Γ(c,λ)v²
            dA/dt = f(c,v,λ)
            """
            # Extract state components
            c = state[:, 0]
            v = state[:, 1]
            A = state[:, 2]
                
            # Get Christoffel symbols via interpolation
            christoffel = self.christoffel_computer.interpolate(c, self._current_wavelengths)
            
            # Compute geodesic derivatives
            dc_dt = v
            dv_dt = -christoffel * v * v
            
            # Compute spectral flow using neural network
            # f(c,v,λ) models how absorbance changes along geodesic
            flow_input = torch.stack([c, v, self._current_wavelengths], dim=1)
            dA_dt = self.spectral_flow_network(flow_input).squeeze(-1)
            
            # Ensure dA_dt has same shape as other derivatives
            if dA_dt.dim() == 0:
                dA_dt = dA_dt.unsqueeze(0)
            
            # Stack all derivatives for coupled system
            derivatives = torch.stack([dc_dt, dv_dt, dA_dt], dim=1)
            
            return derivatives
            
        # Choose integration method
        # Force non-adjoint for MPS compatibility
        if False and self.use_adjoint and initial_states.requires_grad:
            # Use adjoint method for memory efficiency during training
            # Collect parameters that need gradients
            adjoint_params = tuple()
            if self.spectral_flow_network is not None:
                adjoint_params = tuple(self.spectral_flow_network.parameters())
            
            trajectories = odeint_adjoint(
                coupled_geodesic_ode,
                initial_states,
                t_span,
                method=method,
                rtol=rtol,
                atol=atol,
                adjoint_params=adjoint_params
            )
        else:
            # Standard integration for inference
            trajectories = odeint(
                coupled_geodesic_ode,
                initial_states,
                t_span,
                method=method,
                rtol=float(rtol),  # Ensure Python float
                atol=float(atol)   # Ensure Python float
            )
            
        # Extract final states
        final_states = trajectories[-1]
        
        # Extract final absorbance (the prediction)
        final_absorbance = final_states[:, 2]
        
        return {
            'trajectories': trajectories,
            'final_states': final_states,
            'final_absorbance': final_absorbance
        }
        
    def parallel_integrate(self,
                          c_sources: torch.Tensor,
                          c_targets: torch.Tensor,
                          wavelengths: torch.Tensor,
                          initial_velocities: torch.Tensor,
                          n_time_points: int = 50) -> Dict[str, torch.Tensor]:
        """
        Convenience method for parallel integration of multiple geodesics
        
        Args:
            c_sources: Source concentrations [batch_size]
            c_targets: Target concentrations [batch_size] (for validation)
            wavelengths: Wavelengths [batch_size]
            initial_velocities: Initial velocities from shooting solver [batch_size]
            n_time_points: Number of time points for trajectory
            
        Returns:
            Integration results dictionary
        """
        batch_size = c_sources.shape[0]
        
        # Prepare initial states with zero initial absorbance
        # A(t=0) = 0 for concentration transitions
        initial_A = torch.zeros(batch_size, device=self.device)
        initial_states = torch.stack([c_sources, initial_velocities, initial_A], dim=1)
            
        # Time span from 0 to 1
        t_span = torch.linspace(0, 1, n_time_points, device=self.device)
        
        # Integrate
        results = self.integrate_batch(
            initial_states=initial_states,
            wavelengths=wavelengths,
            t_span=t_span
        )
        
        # Add endpoint validation
        c_final = results['final_states'][:, 0]
        c_endpoint_error = (c_final - c_targets).abs()
        results['c_endpoint_error'] = c_endpoint_error
        results['mean_c_error'] = c_endpoint_error.mean()
        
        return results