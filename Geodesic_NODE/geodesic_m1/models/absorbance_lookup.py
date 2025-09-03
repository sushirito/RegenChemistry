"""
Absorbance lookup network for initial conditions
Provides actual absorbance values at source concentrations
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class AbsorbanceLookup(nn.Module):
    """
    Network that provides absorbance values for any concentration/wavelength pair
    Uses actual data at training points, learned interpolation elsewhere
    """
    
    def __init__(self, 
                 concentrations: np.ndarray,
                 wavelengths: np.ndarray,
                 absorbance_matrix: np.ndarray,
                 device: torch.device = torch.device('cpu')):
        """
        Args:
            concentrations: Training concentration values [n_concs]
            wavelengths: Wavelength values [n_wavelengths]
            absorbance_matrix: Actual absorbance data [n_wavelengths, n_concs] or [n_concs, n_wavelengths]
            device: Computation device
        """
        super().__init__()
        
        # Ensure absorbance_matrix is in correct shape [n_concs, n_wavelengths]
        if absorbance_matrix.shape[0] == len(wavelengths) and absorbance_matrix.shape[1] == len(concentrations):
            # Matrix is [n_wavelengths, n_concs], need to transpose
            absorbance_matrix = absorbance_matrix.T
        
        # Store training data as buffers (no gradients needed)
        self.register_buffer('train_concs', torch.tensor(concentrations, dtype=torch.float32))
        self.register_buffer('train_wavelengths', torch.tensor(wavelengths, dtype=torch.float32))
        self.register_buffer('abs_matrix', torch.tensor(absorbance_matrix, dtype=torch.float32))
        
        # Normalization parameters
        self.c_mean, self.c_std = 30.0, 30.0
        self.wl_mean, self.wl_std = 500.0, 300.0
        self.A_mean = float(absorbance_matrix.mean())
        self.A_std = float(absorbance_matrix.std())
        
        # Learnable smooth interpolation network
        self.interpolation_network = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        self.device = device
        self.to(device)
        
        # Initialize network to roughly match data statistics
        self._initialize_network()
        
    def _initialize_network(self):
        """Initialize network weights for stable predictions"""
        for module in self.interpolation_network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
        # Set output bias to match mean absorbance
        with torch.no_grad():
            last_layer = self.interpolation_network[-1]
            last_layer.bias.fill_(0.0)  # Will output normalized values
    
    def lookup_exact(self, c_norm: torch.Tensor, wl_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Look up exact values from training data if available
        
        Returns:
            values: Exact absorbance values where available
            mask: Boolean mask indicating which values were found
        """
        batch_size = c_norm.shape[0]
        values = torch.zeros(batch_size, device=self.device)
        mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        # Denormalize inputs
        c = c_norm * self.c_std + self.c_mean
        wl = wl_norm * self.wl_std + self.wl_mean
        
        # Check each point
        for i in range(batch_size):
            # Find matching concentration (within tolerance)
            c_matches = torch.abs(self.train_concs - c[i]) < 0.1
            
            if c_matches.any():
                c_idx = torch.where(c_matches)[0][0]
                
                # Find matching wavelength
                wl_matches = torch.abs(self.train_wavelengths - wl[i]) < 0.1
                
                if wl_matches.any():
                    wl_idx = torch.where(wl_matches)[0][0]
                    # Get exact value
                    exact_value = self.abs_matrix[c_idx, wl_idx]
                    # Normalize it
                    values[i] = (exact_value - self.A_mean) / self.A_std
                    mask[i] = True
                    
        return values, mask
    
    def forward(self, c_norm: torch.Tensor, wl_norm: torch.Tensor) -> torch.Tensor:
        """
        Get absorbance values for given concentration/wavelength pairs
        
        Args:
            c_norm: Normalized concentrations [batch_size]
            wl_norm: Normalized wavelengths [batch_size]
            
        Returns:
            Normalized absorbance values [batch_size]
        """
        batch_size = c_norm.shape[0]
        
        # Look up exact values where available
        exact_values, exact_mask = self.lookup_exact(c_norm, wl_norm)
        
        # For remaining points, use interpolation network
        if not exact_mask.all():
            # Prepare input
            inputs = torch.stack([c_norm, wl_norm], dim=-1)  # [batch_size, 2]
            
            # Get interpolated values
            interp_values = self.interpolation_network(inputs).squeeze(-1)
            
            # Combine exact and interpolated
            result = torch.where(exact_mask, exact_values, interp_values)
        else:
            result = exact_values
            
        return result
    
    def get_source_absorbance(self, c_source: torch.Tensor, wavelength: torch.Tensor) -> torch.Tensor:
        """
        Get absorbance at source concentrations (for initial conditions)
        
        Args:
            c_source: Source concentrations (normalized) [batch_size]
            wavelength: Wavelengths (normalized) [batch_size]
            
        Returns:
            Initial absorbance values (normalized) [batch_size]
        """
        return self.forward(c_source, wavelength)