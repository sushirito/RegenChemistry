"""
Multi-model ensemble for leave-one-out validation
Each model excludes one concentration during training
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

from .geodesic_model import GeodesicNODE


class MultiModelEnsemble(nn.Module):
    """Ensemble of 6 Geodesic NODE models for leave-one-out validation"""
    
    def __init__(self,
                 num_models: int = 6,
                 metric_hidden_dims: list = None,
                 flow_hidden_dims: list = None,
                 n_trajectory_points: int = 50,
                 shooting_max_iter: int = 10,
                 shooting_tolerance: float = 1e-4,
                 christoffel_grid_size: tuple = (2000, 601),
                 device: torch.device = None,
                 share_metric: bool = True):
        """
        Initialize ensemble of geodesic models
        
        Args:
            num_models: Number of models (6 for leave-one-out)
            metric_hidden_dims: Hidden dims for metric network
            flow_hidden_dims: Hidden dims for spectral flow network
            n_trajectory_points: Points for trajectory discretization
            shooting_max_iter: Max iterations for BVP solver
            shooting_tolerance: Convergence tolerance for shooting
            christoffel_grid_size: Grid resolution for Christoffel symbols
            device: Computation device
            share_metric: Whether all models share the same metric network
        """
        super().__init__()
        
        self.num_models = num_models
        self.share_metric = share_metric
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # Create models
        if share_metric:
            # Single shared metric network
            self.shared_model = GeodesicNODE(
                metric_hidden_dims=metric_hidden_dims,
                flow_hidden_dims=flow_hidden_dims,
                n_trajectory_points=n_trajectory_points,
                shooting_max_iter=shooting_max_iter,
                shooting_tolerance=shooting_tolerance,
                christoffel_grid_size=christoffel_grid_size,
                device=device
            )
            
            # Create additional spectral flow networks for each model
            from .spectral_flow_network import SpectralFlowNetwork
            self.flow_networks = nn.ModuleList([
                SpectralFlowNetwork(
                    input_dim=3,
                    hidden_dims=flow_hidden_dims,
                    activation='tanh'
                ).to(device)
                for _ in range(num_models)
            ])
            
        else:
            # Independent models
            self.models = nn.ModuleList([
                GeodesicNODE(
                    metric_hidden_dims=metric_hidden_dims,
                    flow_hidden_dims=flow_hidden_dims,
                    n_trajectory_points=n_trajectory_points,
                    shooting_max_iter=shooting_max_iter,
                    shooting_tolerance=shooting_tolerance,
                    christoffel_grid_size=christoffel_grid_size,
                    device=device
                )
                for _ in range(num_models)
            ])
            
        # Track which concentration each model excludes
        self.excluded_concentrations = list(range(num_models))
        
    def precompute_christoffel_grids(self, c_range: tuple = (-1, 1),
                                    lambda_range: tuple = (-1, 1)):
        """Pre-compute Christoffel grids for all models"""
        if self.share_metric:
            self.shared_model.precompute_christoffel_grid(c_range, lambda_range)
        else:
            for model in self.models:
                model.precompute_christoffel_grid(c_range, lambda_range)
                
    def forward(self, model_idx: int, c_sources: torch.Tensor,
                c_targets: torch.Tensor, wavelengths: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through specific model
        
        Args:
            model_idx: Which model to use (0-5)
            c_sources: Source concentrations [batch_size]
            c_targets: Target concentrations [batch_size]
            wavelengths: Wavelengths [batch_size]
            
        Returns:
            Model predictions dictionary
        """
        if self.share_metric:
            # Temporarily replace spectral flow network
            original_flow = self.shared_model.spectral_flow_network
            self.shared_model.spectral_flow_network = self.flow_networks[model_idx]
            
            # Also update in integrator
            self.shared_model.geodesic_integrator.spectral_flow_network = self.flow_networks[model_idx]
            
            # Forward pass
            results = self.shared_model(c_sources, c_targets, wavelengths)
            
            # Restore original
            self.shared_model.spectral_flow_network = original_flow
            self.shared_model.geodesic_integrator.spectral_flow_network = original_flow
            
            return results
        else:
            return self.models[model_idx](c_sources, c_targets, wavelengths)
            
    def get_model(self, idx: int) -> GeodesicNODE:
        """Get specific model"""
        if self.share_metric:
            # Return shared model with specific flow network
            # Note: This is a simplified view, actual usage should go through forward()
            return self.shared_model
        else:
            return self.models[idx]
            
    def train_model(self, model_idx: int, mode: bool = True):
        """Set training mode for specific model"""
        if self.share_metric:
            self.shared_model.train(mode)
            self.flow_networks[model_idx].train(mode)
        else:
            self.models[model_idx].train(mode)
            
    def eval_model(self, model_idx: int):
        """Set eval mode for specific model"""
        self.train_model(model_idx, mode=False)
        
    def parameters_for_model(self, model_idx: int):
        """Get parameters for specific model (for optimizer)"""
        if self.share_metric:
            # Return both metric and specific flow network parameters
            return list(self.shared_model.metric_network.parameters()) + \
                   list(self.flow_networks[model_idx].parameters())
        else:
            return self.models[model_idx].parameters()