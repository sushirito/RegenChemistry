#!/usr/bin/env python3
"""
Loss Functions for Geodesic Spectral Model
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class GeodesicLoss(nn.Module):
    """
    Combined loss function for geodesic spectral model
    """
    
    def __init__(self,
                 reconstruction_weight: float = 1.0,
                 smoothness_weight: float = 0.01,
                 bounds_weight: float = 0.001,
                 path_weight: float = 0.001):
        """
        Initialize loss function
        
        Args:
            reconstruction_weight: Weight for MSE reconstruction loss
            smoothness_weight: Weight for metric smoothness regularization
            bounds_weight: Weight for metric bounds constraint
            path_weight: Weight for path length regularization
        """
        super().__init__()
        
        self.reconstruction_weight = reconstruction_weight
        self.smoothness_weight = smoothness_weight
        self.bounds_weight = bounds_weight
        self.path_weight = path_weight
        
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
               metric_network: Optional[nn.Module] = None,
               path_lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute total loss and individual components
        
        Args:
            predictions: Predicted absorbance values
            targets: Target absorbance values
            metric_network: Metric network for regularization
            path_lengths: Geodesic path lengths for regularization
        
        Returns:
            Dictionary with total loss and components
        """
        losses = {}
        
        # Main reconstruction loss
        losses['reconstruction'] = self.mse_loss(predictions, targets)
        
        # Total loss starts with reconstruction
        total_loss = self.reconstruction_weight * losses['reconstruction']
        
        # Metric smoothness regularization
        if metric_network is not None and self.smoothness_weight > 0:
            losses['smoothness'] = self._compute_smoothness_loss(metric_network)
            total_loss += self.smoothness_weight * losses['smoothness']
        
        # Metric bounds constraint
        if metric_network is not None and self.bounds_weight > 0:
            losses['bounds'] = self._compute_bounds_loss(metric_network)
            total_loss += self.bounds_weight * losses['bounds']
        
        # Path length regularization
        if path_lengths is not None and self.path_weight > 0:
            losses['path_length'] = path_lengths.mean()
            total_loss += self.path_weight * losses['path_length']
        
        losses['total'] = total_loss
        
        return losses
    
    def _compute_smoothness_loss(self, metric_network: nn.Module) -> torch.Tensor:
        """
        Compute smoothness regularization for metric network
        """
        # Sample random points
        device = next(metric_network.parameters()).device
        n_samples = 100
        
        c = torch.randn(n_samples, device=device)
        wl = torch.randn(n_samples, device=device)
        
        # Compute metric
        g = metric_network(c, wl)
        
        # Smoothness: penalize large second derivatives
        epsilon = 1e-3
        g_plus = metric_network(c + epsilon, wl)
        g_minus = metric_network(c - epsilon, wl)
        
        second_derivative = (g_plus - 2*g + g_minus) / (epsilon**2)
        smoothness_loss = second_derivative.abs().mean()
        
        return smoothness_loss
    
    def _compute_bounds_loss(self, metric_network: nn.Module) -> torch.Tensor:
        """
        Keep metric within reasonable bounds
        """
        # Sample random points
        device = next(metric_network.parameters()).device
        n_samples = 100
        
        c = torch.randn(n_samples, device=device)
        wl = torch.randn(n_samples, device=device)
        
        # Compute metric
        g = metric_network(c, wl)
        
        # Penalize values outside [0.01, 100]
        lower_bound_penalty = torch.relu(0.01 - g).mean()
        upper_bound_penalty = torch.relu(g - 100).mean()
        
        bounds_loss = lower_bound_penalty + upper_bound_penalty
        
        return bounds_loss


class SpectralSimilarityLoss(nn.Module):
    """
    Additional loss based on spectral similarity metrics
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral similarity loss
        """
        # Normalize spectra
        pred_norm = predictions / (predictions.norm(dim=-1, keepdim=True) + 1e-8)
        target_norm = targets / (targets.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Cosine similarity
        cosine_sim = (pred_norm * target_norm).sum(dim=-1)
        
        # Loss is 1 - similarity (so perfect match = 0 loss)
        loss = 1 - cosine_sim.mean()
        
        return loss


class WeightedMSELoss(nn.Module):
    """
    MSE loss with wavelength-dependent weighting
    """
    
    def __init__(self, wavelength_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.wavelength_weights = wavelength_weights
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
               wavelength_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute weighted MSE loss
        """
        squared_errors = (predictions - targets).square()
        
        if self.wavelength_weights is not None and wavelength_idx is not None:
            weights = self.wavelength_weights[wavelength_idx]
            weighted_errors = squared_errors * weights
            loss = weighted_errors.mean()
        else:
            loss = squared_errors.mean()
        
        return loss


def test_losses():
    """Test loss functions"""
    print("Testing Loss Functions...")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create test data
    batch_size = 256
    predictions = torch.randn(batch_size, device=device) * 0.1
    targets = torch.randn(batch_size, device=device) * 0.1
    
    # Test geodesic loss
    print("\nTesting GeodesicLoss...")
    geodesic_loss = GeodesicLoss()
    
    losses = geodesic_loss(predictions, targets)
    print(f"Reconstruction loss: {losses['reconstruction'].item():.6f}")
    print(f"Total loss: {losses['total'].item():.6f}")
    
    # Test with metric network
    from geodesic_mps.models.metric_network import OptimizedMetricNetwork
    metric = OptimizedMetricNetwork().to(device)
    
    path_lengths = torch.rand(batch_size, device=device) * 2
    
    losses_full = geodesic_loss(predictions, targets, metric, path_lengths)
    print(f"\nWith regularization:")
    for key, value in losses_full.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.6f}")
    
    # Test spectral similarity loss
    print("\nTesting SpectralSimilarityLoss...")
    spectral_loss = SpectralSimilarityLoss()
    
    # Create spectral data (batch, wavelengths)
    pred_spectra = torch.randn(32, 100, device=device)
    target_spectra = torch.randn(32, 100, device=device)
    
    sim_loss = spectral_loss(pred_spectra, target_spectra)
    print(f"Spectral similarity loss: {sim_loss.item():.6f}")
    
    # Test weighted MSE
    print("\nTesting WeightedMSELoss...")
    wavelength_weights = torch.ones(601, device=device)
    wavelength_weights[250:350] = 2.0  # Emphasize important region
    
    weighted_loss = WeightedMSELoss(wavelength_weights)
    wl_idx = torch.randint(0, 601, (batch_size,), device=device)
    
    w_loss = weighted_loss(predictions, targets, wl_idx)
    print(f"Weighted MSE loss: {w_loss.item():.6f}")
    
    # Test gradient flow
    print("\nTesting gradient flow...")
    total_loss = losses_full['total'] + sim_loss + w_loss
    total_loss.backward()
    
    has_grad = all(p.grad is not None for p in metric.parameters() if p.requires_grad)
    print(f"Metric network has gradients: {has_grad}")
    
    print("\nLoss tests passed!")


if __name__ == "__main__":
    test_losses()