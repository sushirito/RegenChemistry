#!/usr/bin/env python3
"""
Full Geodesic Spectral Model
Integrates all components: Metric → Christoffel → ODE → Shooting → Decoder
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from src.models.metric_network import MetricNetwork, compute_metric_smoothness_loss, compute_metric_bounds_loss
from src.core.christoffel import ChristoffelComputer
from src.core.geodesic_ode import GeodesicODE
from src.core.shooting_solver import ShootingSolver
from src.models.decoder_network import AbsorbanceDecoder


class GeodesicSpectralModel(nn.Module):
    """
    End-to-end model for spectral interpolation using geodesics
    
    Pipeline:
    1. Learn metric g(c,λ) defining the geometry
    2. Compute Christoffel symbols Γ(c,λ)
    3. Solve geodesic BVP to connect concentrations
    4. Extract path statistics from geodesic
    5. Decode to absorbance prediction
    """
    
    def __init__(self, 
                 n_trajectory_points: int = 11,
                 shooting_tolerance: float = 1e-4,
                 shooting_max_iter: int = 50):
        """
        Initialize the full model
        
        Args:
            n_trajectory_points: Number of points along geodesic path
            shooting_tolerance: Tolerance for BVP solver
            shooting_max_iter: Maximum iterations for shooting method
        """
        super(GeodesicSpectralModel, self).__init__()
        
        # Core components
        self.metric_network = MetricNetwork()
        self.decoder = AbsorbanceDecoder()
        
        # Non-learnable components
        self.christoffel_computer = ChristoffelComputer()
        self.shooting_solver = ShootingSolver(
            self.metric_network,
            max_iterations=shooting_max_iter,
            tolerance=shooting_tolerance
        )
        
        # Configuration
        self.n_trajectory_points = n_trajectory_points
        
        # Statistics tracking
        self.forward_stats = {
            'total_forward': 0,
            'successful_geodesics': 0,
            'failed_geodesics': 0
        }
    
    def solve_geodesic_bvp(self,
                          c_source: float,
                          c_target: float,
                          wavelength: float) -> Dict:
        """
        Solve the boundary value problem for a single geodesic
        
        Args:
            c_source: Starting concentration (normalized)
            c_target: Target concentration (normalized)
            wavelength: Wavelength (normalized)
        
        Returns:
            Solution dictionary from shooting solver
        """
        return self.shooting_solver.solve(
            c_source, c_target, wavelength,
            return_trajectory=True,
            n_trajectory_points=self.n_trajectory_points
        )
    
    def forward(self,
                c_source: torch.Tensor,
                c_target: torch.Tensor,
                wavelength: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the full model
        
        Args:
            c_source: Source concentrations (batch_size,)
            c_target: Target concentrations (batch_size,)
            wavelength: Wavelengths (batch_size,)
        
        Returns:
            Dictionary containing:
                - absorbance: Predicted absorbance values
                - trajectories: Geodesic paths
                - success_mask: Which geodesics were successfully solved
        """
        self.forward_stats['total_forward'] += 1
        batch_size = c_source.shape[0]
        
        # Solve geodesic BVPs for each sample
        trajectories = []
        success_list = []
        
        for i in range(batch_size):
            result = self.solve_geodesic_bvp(
                c_source[i].item(),
                c_target[i].item(),
                wavelength[i].item()
            )
            
            if result['success']:
                trajectories.append(torch.tensor(result['trajectory'], dtype=torch.float32))
                success_list.append(True)
                self.forward_stats['successful_geodesics'] += 1
            else:
                # Use linear interpolation as fallback
                t = np.linspace(0, 1, self.n_trajectory_points)
                c_linear = c_source[i].item() + t * (c_target[i].item() - c_source[i].item())
                v_linear = np.full(self.n_trajectory_points, c_target[i].item() - c_source[i].item())
                fallback_traj = np.stack([c_linear, v_linear], axis=1)
                trajectories.append(torch.tensor(fallback_traj, dtype=torch.float32))
                success_list.append(False)
                self.forward_stats['failed_geodesics'] += 1
        
        # Stack trajectories
        trajectories = torch.stack(trajectories)  # (batch_size, n_points, 2)
        success_mask = torch.tensor(success_list, dtype=torch.bool)
        
        # Decode trajectories to absorbance
        absorbance = self.decoder(trajectories, wavelength)
        
        return {
            'absorbance': absorbance,
            'trajectories': trajectories,
            'success_mask': success_mask
        }
    
    def compute_loss(self,
                    predictions: Dict[str, torch.Tensor],
                    targets: torch.Tensor,
                    c_samples: Optional[torch.Tensor] = None,
                    wavelength_samples: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components
        
        Args:
            predictions: Output from forward pass
            targets: Target absorbance values
            c_samples: Concentration samples for regularization
            wavelength_samples: Wavelength samples for regularization
        
        Returns:
            Dictionary of loss components
        """
        # Reconstruction loss (main objective)
        loss_recon = nn.functional.mse_loss(predictions['absorbance'], targets)
        
        losses = {'reconstruction': loss_recon}
        
        # Regularization losses (if samples provided)
        if c_samples is not None and wavelength_samples is not None:
            # Metric smoothness
            loss_smooth = compute_metric_smoothness_loss(
                self.metric_network, c_samples, wavelength_samples
            )
            losses['smoothness'] = loss_smooth
            
            # Metric bounds
            loss_bounds = compute_metric_bounds_loss(
                self.metric_network, c_samples, wavelength_samples
            )
            losses['bounds'] = loss_bounds
            
            # Path efficiency (minimize path length)
            trajectories = predictions['trajectories']
            path_lengths = self.decoder.compute_path_length(trajectories)
            loss_path = path_lengths.mean()
            losses['path_efficiency'] = loss_path
        
        return losses
    
    def combine_losses(self,
                      losses: Dict[str, torch.Tensor],
                      weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """
        Combine loss components with weights
        
        Args:
            losses: Dictionary of loss components
            weights: Loss weights (uses defaults if None)
        
        Returns:
            Total weighted loss
        """
        if weights is None:
            weights = {
                'reconstruction': 1.0,
                'smoothness': 0.01,
                'bounds': 0.001,
                'path_efficiency': 0.001
            }
        
        total_loss = torch.tensor(0.0, dtype=torch.float32)
        for key, loss in losses.items():
            weight = weights.get(key, 0.0)
            total_loss = total_loss + weight * loss
        
        return total_loss
    
    def get_stats(self) -> Dict:
        """Get model statistics"""
        stats = self.forward_stats.copy()
        stats.update(self.shooting_solver.get_stats())
        return stats


def test_geodesic_model():
    """Test the integrated model"""
    print("Testing Geodesic Spectral Model...")
    
    # Create model
    model = GeodesicSpectralModel()
    
    # Test single forward pass
    print("\n1. Single forward pass:")
    c_source = torch.tensor([0.0])
    c_target = torch.tensor([0.5])
    wavelength = torch.tensor([0.0])
    
    output = model(c_source, c_target, wavelength)
    print(f"  Absorbance: {output['absorbance'].item():.4f}")
    print(f"  Trajectory shape: {output['trajectories'].shape}")
    print(f"  Success: {output['success_mask'].item()}")
    
    # Test batch forward pass
    print("\n2. Batch forward pass:")
    batch_size = 4
    c_sources = torch.randn(batch_size) * 0.5
    c_targets = torch.randn(batch_size) * 0.5
    wavelengths = torch.randn(batch_size) * 0.5
    
    outputs = model(c_sources, c_targets, wavelengths)
    print(f"  Batch size: {batch_size}")
    print(f"  Absorbances shape: {outputs['absorbance'].shape}")
    print(f"  Success rate: {outputs['success_mask'].float().mean().item():.1%}")
    
    # Test loss computation
    print("\n3. Loss computation:")
    targets = torch.randn(batch_size)
    c_samples = torch.randn(10)
    wavelength_samples = torch.randn(10)
    
    losses = model.compute_loss(outputs, targets, c_samples, wavelength_samples)
    print("  Loss components:")
    for key, value in losses.items():
        print(f"    {key}: {value.item():.6f}")
    
    total_loss = model.combine_losses(losses)
    print(f"  Total loss: {total_loss.item():.6f}")
    
    # Test gradient flow
    print("\n4. Gradient flow test:")
    model.zero_grad()
    total_loss.backward()
    
    # Check gradients in both networks
    metric_has_grad = any(p.grad is not None and p.grad.abs().max() > 0 
                          for p in model.metric_network.parameters())
    decoder_has_grad = any(p.grad is not None and p.grad.abs().max() > 0 
                           for p in model.decoder.parameters())
    
    print(f"  Metric network gradients: {'✓' if metric_has_grad else '✗'}")
    print(f"  Decoder network gradients: {'✓' if decoder_has_grad else '✗'}")
    
    # Parameter count
    print("\n5. Model complexity:")
    n_metric = sum(p.numel() for p in model.metric_network.parameters())
    n_decoder = sum(p.numel() for p in model.decoder.parameters())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Metric network: {n_metric:,} parameters")
    print(f"  Decoder network: {n_decoder:,} parameters")
    print(f"  Total: {n_total:,} parameters")
    
    # Statistics
    print("\n6. Model statistics:")
    stats = model.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nGeodesic Model tests completed!")


if __name__ == "__main__":
    test_geodesic_model()