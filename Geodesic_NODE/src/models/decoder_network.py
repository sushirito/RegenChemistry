#!/usr/bin/env python3
"""
Absorbance Decoder Network with Path Statistics
Maps geodesic path features to absorbance predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class AbsorbanceDecoder(nn.Module):
    """
    Decoder network that uses geodesic path statistics to predict absorbance
    
    Uses Option B from the specification: Path Statistics
    Input: [c_final, c_mean, path_length, max_velocity, λ]
    Architecture: 5 → 64 → 128 → 1
    """
    
    def __init__(self):
        super(AbsorbanceDecoder, self).__init__()
        
        # Architecture as specified: 5 → 64 → 128 → 1
        self.decoder = nn.Sequential(
            nn.Linear(5, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def compute_path_length(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Compute geodesic path length from trajectory
        
        Args:
            trajectory: Geodesic path (batch_size, n_points, 2) or (n_points, 2)
        
        Returns:
            Path length (batch_size,) or scalar
        """
        if trajectory.dim() == 2:
            # Single trajectory
            c_path = trajectory[:, 0]
            # Compute differences between consecutive points
            dc = torch.diff(c_path)
            # Sum absolute differences (approximation of arc length)
            path_length = torch.sum(torch.abs(dc))
        else:
            # Batch of trajectories
            c_paths = trajectory[:, :, 0]  # (batch_size, n_points)
            # Compute differences
            dc = torch.diff(c_paths, dim=1)  # (batch_size, n_points-1)
            # Sum along path dimension
            path_length = torch.sum(torch.abs(dc), dim=1)  # (batch_size,)
        
        return path_length
    
    def extract_path_statistics(self, 
                               trajectory: torch.Tensor,
                               wavelength: torch.Tensor) -> torch.Tensor:
        """
        Extract statistics from geodesic trajectory
        
        Args:
            trajectory: Geodesic path (batch_size, n_points, 2) or (n_points, 2)
                       where dim 2 is [c, v]
            wavelength: Wavelength values (batch_size,) or scalar
        
        Returns:
            Feature vector [c_final, c_mean, path_length, max_velocity, λ]
        """
        # Handle single trajectory
        if trajectory.dim() == 2:
            trajectory = trajectory.unsqueeze(0)
            single_traj = True
        else:
            single_traj = False
        
        # Handle single wavelength
        if wavelength.dim() == 0 or wavelength.shape[0] == 1:
            wavelength = wavelength.expand(trajectory.shape[0])
        
        # Extract concentration and velocity paths
        c_paths = trajectory[:, :, 0]  # (batch_size, n_points)
        v_paths = trajectory[:, :, 1]  # (batch_size, n_points)
        
        # Compute statistics
        c_final = c_paths[:, -1]  # Final concentration
        c_mean = torch.mean(c_paths, dim=1)  # Mean concentration along path
        path_length = self.compute_path_length(trajectory)  # Path length
        max_velocity = torch.max(torch.abs(v_paths), dim=1)[0]  # Maximum absolute velocity
        
        # Stack features
        features = torch.stack([
            c_final,
            c_mean,
            path_length,
            max_velocity,
            wavelength
        ], dim=1)  # (batch_size, 5)
        
        if single_traj:
            features = features.squeeze(0)
        
        return features
    
    def forward(self, 
                trajectory: torch.Tensor,
                wavelength: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: trajectory → statistics → absorbance
        
        Args:
            trajectory: Geodesic path (batch_size, n_points, 2) or (n_points, 2)
            wavelength: Wavelength values (batch_size,) or scalar
        
        Returns:
            Predicted absorbance (batch_size,) or scalar
        """
        # Extract path statistics
        features = self.extract_path_statistics(trajectory, wavelength)
        
        # Decode to absorbance
        absorbance = self.decoder(features)
        
        # Remove last dimension if needed
        if absorbance.dim() > 1:
            absorbance = absorbance.squeeze(-1)
        
        return absorbance


class SimpleDecoder(nn.Module):
    """
    Simple decoder using only endpoint (Option A from specification)
    For comparison and debugging
    
    Input: [c_final, λ]
    Architecture: 2 → 64 → 128 → 1
    """
    
    def __init__(self):
        super(SimpleDecoder, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, 
                c_final: torch.Tensor,
                wavelength: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: (c_final, λ) → absorbance
        """
        if c_final.dim() == 0:
            c_final = c_final.unsqueeze(0)
        if wavelength.dim() == 0:
            wavelength = wavelength.unsqueeze(0)
        
        features = torch.stack([c_final, wavelength], dim=-1)
        absorbance = self.decoder(features).squeeze(-1)
        
        return absorbance


def test_decoder_networks():
    """Test both decoder implementations"""
    print("Testing Decoder Networks...")
    
    # Test AbsorbanceDecoder with path statistics
    print("\n1. Testing AbsorbanceDecoder (Path Statistics):")
    decoder = AbsorbanceDecoder()
    
    # Single trajectory test
    n_points = 11
    trajectory = torch.randn(n_points, 2)  # Random geodesic path
    wavelength = torch.tensor(0.5)
    
    absorbance = decoder(trajectory, wavelength)
    print(f"  Single trajectory: input shape {trajectory.shape} → output {absorbance.item():.4f}")
    
    # Batch test
    batch_size = 8
    trajectories = torch.randn(batch_size, n_points, 2)
    wavelengths = torch.randn(batch_size)
    
    absorbances = decoder(trajectories, wavelengths)
    print(f"  Batch: input shape {trajectories.shape} → output shape {absorbances.shape}")
    
    # Test path statistics extraction
    features = decoder.extract_path_statistics(trajectories, wavelengths)
    print(f"  Extracted features shape: {features.shape}")
    print(f"  Feature names: [c_final, c_mean, path_length, max_velocity, λ]")
    print(f"  Sample features: {features[0]}")
    
    # Test gradient flow
    loss = absorbances.mean()
    loss.backward()
    
    grad_exists = all(p.grad is not None for p in decoder.parameters())
    print(f"  Gradient flow: {'✓' if grad_exists else '✗'}")
    
    # Test SimpleDecoder
    print("\n2. Testing SimpleDecoder (Endpoint Only):")
    simple_decoder = SimpleDecoder()
    
    c_finals = torch.randn(batch_size)
    wavelengths = torch.randn(batch_size)
    
    absorbances_simple = simple_decoder(c_finals, wavelengths)
    print(f"  Batch: input shapes {c_finals.shape}, {wavelengths.shape} → output {absorbances_simple.shape}")
    
    # Compare parameter counts
    n_params_full = sum(p.numel() for p in decoder.parameters())
    n_params_simple = sum(p.numel() for p in simple_decoder.parameters())
    print(f"\n3. Model Complexity:")
    print(f"  AbsorbanceDecoder parameters: {n_params_full:,}")
    print(f"  SimpleDecoder parameters: {n_params_simple:,}")
    
    # Test robustness to different trajectory lengths
    print("\n4. Testing variable trajectory lengths:")
    for n_pts in [5, 11, 21, 51]:
        traj = torch.randn(n_pts, 2)
        wl = torch.tensor(0.0)
        try:
            abs_val = decoder(traj, wl)
            print(f"  {n_pts} points: ✓ (output = {abs_val.item():.4f})")
        except Exception as e:
            print(f"  {n_pts} points: ✗ ({e})")
    
    print("\nDecoder Network tests completed!")


if __name__ == "__main__":
    test_decoder_networks()