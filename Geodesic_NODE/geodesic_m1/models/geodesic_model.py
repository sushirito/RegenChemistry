"""
End-to-end Geodesic-Coupled Spectral NODE model
Integrates metric network, spectral flow, and geodesic solver
Optimized for M1 Mac MPS acceleration
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple

from geodesic_m1.core.christoffel_computer import ChristoffelComputer
from geodesic_m1.core.geodesic_integrator import GeodesicIntegrator
from geodesic_m1.core.shooting_solver import ShootingSolver
from geodesic_m1.models.metric_network import MetricNetwork
from geodesic_m1.models.spectral_flow_network import SpectralFlowNetwork
from geodesic_m1.models.absorbance_lookup import AbsorbanceLookup


class GeodesicNODE(nn.Module):
    """Complete Geodesic-Coupled Neural ODE system for spectral prediction"""
    
    def __init__(self,
                 metric_hidden_dims: list = None,
                 flow_hidden_dims: list = None,
                 n_trajectory_points: int = 50,
                 shooting_max_iter: int = 10,
                 shooting_tolerance: float = 1e-4,
                 shooting_learning_rate: float = 0.5,
                 christoffel_grid_size: tuple = (2000, 601),
                 device: torch.device = None,
                 use_adjoint: bool = True,
                 concentrations: Optional[np.ndarray] = None,
                 wavelengths: Optional[np.ndarray] = None,
                 absorbance_matrix: Optional[np.ndarray] = None):
        """
        Initialize complete geodesic NODE system
        
        Args:
            metric_hidden_dims: Hidden dims for metric network
            flow_hidden_dims: Hidden dims for spectral flow network
            n_trajectory_points: Points for trajectory discretization
            shooting_max_iter: Max iterations for BVP solver
            shooting_tolerance: Convergence tolerance for shooting
            shooting_learning_rate: Learning rate for shooting method
            christoffel_grid_size: Grid resolution for Christoffel symbols
            device: Computation device (MPS for M1 Mac)
            use_adjoint: Use adjoint method for backprop
        """
        super().__init__()
        
        if device is None:
            device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.device = device
        
        # Initialize neural networks
        self.metric_network = MetricNetwork(
            input_dim=2,
            hidden_dims=metric_hidden_dims,
            activation='tanh'
        ).to(device)
        
        self.spectral_flow_network = SpectralFlowNetwork(
            input_dim=3,
            hidden_dims=flow_hidden_dims,
            activation='tanh'
        ).to(device)
        
        # Initialize absorbance lookup (if data provided)
        if concentrations is not None and wavelengths is not None and absorbance_matrix is not None:
            self.absorbance_lookup = AbsorbanceLookup(
                concentrations=concentrations,
                wavelengths=wavelengths,
                absorbance_matrix=absorbance_matrix,
                device=device
            )
        else:
            self.absorbance_lookup = None
        
        # Initialize mathematical components
        self.christoffel_computer = ChristoffelComputer(
            metric_network=self.metric_network,
            grid_size=christoffel_grid_size,
            device=device,
            use_half_precision=True
        )
        
        self.geodesic_integrator = GeodesicIntegrator(
            christoffel_computer=self.christoffel_computer,
            spectral_flow_network=self.spectral_flow_network,
            device=device,
            use_adjoint=use_adjoint
        )
        
        self.shooting_solver = ShootingSolver(
            geodesic_integrator=self.geodesic_integrator,
            absorbance_lookup=self.absorbance_lookup,
            max_iterations=shooting_max_iter,
            tolerance=shooting_tolerance,
            learning_rate=shooting_learning_rate,
            device=device
        )
        
        # Configuration
        self.n_trajectory_points = n_trajectory_points
        self.christoffel_grid_computed = False
        
    def precompute_christoffel_grid(self, c_range: tuple = (-1, 1), 
                                   lambda_range: tuple = (-1, 1)):
        """
        Pre-compute Christoffel symbols on grid for efficiency
        
        Args:
            c_range: Normalized concentration range
            lambda_range: Normalized wavelength range
        """
        self.christoffel_computer.precompute_grid(c_range, lambda_range)
        self.christoffel_grid_computed = True
        
    def forward(self, c_sources: torch.Tensor, c_targets: torch.Tensor,
                wavelengths: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through geodesic NODE system
        
        Args:
            c_sources: Source concentrations [batch_size]
            c_targets: Target concentrations [batch_size]
            wavelengths: Wavelengths [batch_size]
            
        Returns:
            Dictionary with:
                - absorbance: Predicted absorbance at target [batch_size]
                - trajectories: Full state evolution [n_points, batch_size, 3]
                - initial_velocities: Optimal v₀ from shooting [batch_size]
                - convergence_mask: Which geodesics converged [batch_size]
        """
        # Ensure Christoffel grid is computed
        if not self.christoffel_grid_computed:
            self.precompute_christoffel_grid()
            
        # Solve BVP to find initial velocities
        bvp_results = self.shooting_solver.solve_batch(
            c_sources=c_sources,
            c_targets=c_targets,
            wavelengths=wavelengths,
            n_trajectory_points=self.n_trajectory_points
        )
        
        # Extract results
        return {
            'absorbance': bvp_results['final_absorbance'],
            'trajectories': bvp_results['trajectories'],
            'initial_velocities': bvp_results['initial_velocities'],
            'convergence_mask': bvp_results['convergence_mask'],
            'convergence_rate': bvp_results['convergence_rate']
        }
        
    def compute_loss(self, predictions: Dict[str, torch.Tensor],
                    targets: torch.Tensor,
                    c_batch: torch.Tensor,
                    wavelength_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute multi-component loss
        
        Args:
            predictions: Model predictions dictionary
            targets: Target absorbance values [batch_size]
            c_batch: Concentration values for regularization [batch_size]
            wavelength_batch: Wavelength values for regularization [batch_size]
            
        Returns:
            Dictionary with loss components
        """
        # Reconstruction loss
        reconstruction_loss = nn.functional.mse_loss(
            predictions['absorbance'], targets
        )
        
        # Metric smoothness regularization
        smoothness_loss = self.metric_network.get_smoothness_loss(
            c_batch, wavelength_batch
        )
        
        # Metric bounds regularization
        bounds_loss = self.metric_network.get_bounds_loss(
            c_batch, wavelength_batch
        )
        
        # Path length regularization (efficiency)
        if 'trajectories' in predictions:
            trajectories = predictions['trajectories']
            velocities = trajectories[:, :, 1]  # [n_points, batch_size]
            path_lengths = velocities.abs().mean(dim=0).mean()
        else:
            path_lengths = torch.tensor(0.0, device=self.device)
            
        # Total loss with weights
        total_loss = (
            reconstruction_loss +
            0.01 * smoothness_loss +
            0.001 * bounds_loss +
            0.001 * path_lengths
        )
        
        return {
            'total': total_loss,
            'reconstruction': reconstruction_loss,
            'smoothness': smoothness_loss,
            'bounds': bounds_loss,
            'path_length': path_lengths
        }
        
    def parallel_forward(self, concentration_pairs: torch.Tensor,
                        wavelength_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process all 18,030 geodesics in parallel
        
        Args:
            concentration_pairs: All concentration transitions [30, 2]
            wavelength_grid: All wavelengths [601]
            
        Returns:
            Complete results for all geodesics
        """
        # Ensure grid is computed
        if not self.christoffel_grid_computed:
            self.precompute_christoffel_grid()
            
        # Use shooting solver's parallel method
        return self.shooting_solver.solve_parallel(
            concentration_pairs, wavelength_grid
        )
        
    def save_checkpoint(self, path: str, epoch: int, optimizers: dict = None, best_loss: float = None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'metric_network_state': self.metric_network.state_dict(),
            'flow_network_state': self.spectral_flow_network.state_dict(),
            'christoffel_grid': self.christoffel_computer.christoffel_grid.cpu() if self.christoffel_computer.christoffel_grid is not None else None,
            'config': {
                'grid_size': self.christoffel_computer.grid_size,
                'n_trajectory_points': self.n_trajectory_points,
                'device': str(self.device)
            }
        }
        
        if optimizers:
            checkpoint['optimizers'] = {k: v.state_dict() for k, v in optimizers.items()}
        
        if best_loss is not None:
            checkpoint['best_loss'] = best_loss
        
        torch.save(checkpoint, path)
        print(f"💾 Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str, load_optimizers: bool = False):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        self.metric_network.load_state_dict(checkpoint['metric_network_state'])
        self.spectral_flow_network.load_state_dict(checkpoint['flow_network_state'])
        
        if 'christoffel_grid' in checkpoint and checkpoint['christoffel_grid'] is not None:
            self.christoffel_computer.christoffel_grid = checkpoint['christoffel_grid'].to(self.device)
            self.christoffel_grid_computed = True
        
        print(f"✅ Model loaded from {path} (epoch {checkpoint['epoch']})")
        
        if load_optimizers and 'optimizers' in checkpoint:
            return checkpoint['optimizers']
        
        return checkpoint.get('epoch', 0)
        
    def get_model_info(self) -> Dict[str, any]:
        """Get comprehensive model information"""
        total_params = sum(p.numel() for p in self.parameters())
        metric_params = sum(p.numel() for p in self.metric_network.parameters())
        flow_params = sum(p.numel() for p in self.spectral_flow_network.parameters())
        
        return {
            'total_parameters': total_params,
            'metric_network_params': metric_params,
            'flow_network_params': flow_params,
            'christoffel_grid_size': self.christoffel_computer.grid_size,
            'n_trajectory_points': self.n_trajectory_points,
            'device': str(self.device),
            'christoffel_grid_computed': self.christoffel_grid_computed
        }