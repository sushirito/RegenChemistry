"""
Batch preprocessing utilities for M1 Mac MPS
Handles data transformations and augmentation
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np


class SpectralPreprocessor:
    """Preprocessing utilities for spectral data on M1 Mac"""
    
    def __init__(self,
                 device: torch.device = torch.device('mps'),
                 normalization_params: Optional[Dict[str, float]] = None):
        """
        Initialize spectral preprocessor
        
        Args:
            device: Target device (MPS for M1)
            normalization_params: Pre-computed normalization parameters
        """
        self.device = device
        self.normalization_params = normalization_params or {}
        
        # Default normalization parameters
        self.default_params = {
            'c_mean': 30.0,
            'c_std': 30.0,
            'lambda_mean': 500.0,
            'lambda_std': 300.0,
            'A_mean': 0.5,
            'A_std': 0.3
        }
        
    def normalize_concentrations(self, concentrations: torch.Tensor) -> torch.Tensor:
        """
        Normalize concentration values to [-1, 1]
        
        Args:
            concentrations: Raw concentration values (ppb)
            
        Returns:
            Normalized concentrations
        """
        c_mean = self.normalization_params.get('c_mean', self.default_params['c_mean'])
        c_std = self.normalization_params.get('c_std', self.default_params['c_std'])
        
        return (concentrations - c_mean) / c_std
        
    def denormalize_concentrations(self, normalized_concentrations: torch.Tensor) -> torch.Tensor:
        """Denormalize concentration values back to ppb"""
        c_mean = self.normalization_params.get('c_mean', self.default_params['c_mean'])
        c_std = self.normalization_params.get('c_std', self.default_params['c_std'])
        
        return normalized_concentrations * c_std + c_mean
        
    def normalize_wavelengths(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """
        Normalize wavelength values to [-1, 1]
        
        Args:
            wavelengths: Raw wavelength values (nm)
            
        Returns:
            Normalized wavelengths
        """
        lambda_mean = self.normalization_params.get('lambda_mean', self.default_params['lambda_mean'])
        lambda_std = self.normalization_params.get('lambda_std', self.default_params['lambda_std'])
        
        return (wavelengths - lambda_mean) / lambda_std
        
    def denormalize_wavelengths(self, normalized_wavelengths: torch.Tensor) -> torch.Tensor:
        """Denormalize wavelength values back to nm"""
        lambda_mean = self.normalization_params.get('lambda_mean', self.default_params['lambda_mean'])
        lambda_std = self.normalization_params.get('lambda_std', self.default_params['lambda_std'])
        
        return normalized_wavelengths * lambda_std + lambda_mean
        
    def normalize_absorbance(self, absorbance: torch.Tensor) -> torch.Tensor:
        """
        Normalize absorbance values using z-score normalization
        
        Args:
            absorbance: Raw absorbance values
            
        Returns:
            Normalized absorbance
        """
        A_mean = self.normalization_params.get('A_mean', self.default_params['A_mean'])
        A_std = self.normalization_params.get('A_std', self.default_params['A_std'])
        
        return (absorbance - A_mean) / A_std
        
    def denormalize_absorbance(self, normalized_absorbance: torch.Tensor) -> torch.Tensor:
        """Denormalize absorbance values back to original scale"""
        A_mean = self.normalization_params.get('A_mean', self.default_params['A_mean'])
        A_std = self.normalization_params.get('A_std', self.default_params['A_std'])
        
        return normalized_absorbance * A_std + A_mean
        
    def batch_normalize(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Normalize entire batch of data
        
        Args:
            batch: Dictionary with batch data
            
        Returns:
            Dictionary with normalized batch data
        """
        normalized_batch = {}
        
        # Normalize each component
        if 'c_source' in batch:
            normalized_batch['c_source'] = self.normalize_concentrations(batch['c_source'])
        if 'c_target' in batch:
            normalized_batch['c_target'] = self.normalize_concentrations(batch['c_target'])
        if 'wavelength' in batch:
            normalized_batch['wavelength'] = self.normalize_wavelengths(batch['wavelength'])
        if 'target_absorbance' in batch:
            normalized_batch['target_absorbance'] = self.normalize_absorbance(batch['target_absorbance'])
            
        # Copy other fields unchanged
        for key, value in batch.items():
            if key not in normalized_batch:
                normalized_batch[key] = value
                
        return normalized_batch
        
    def add_noise(self, 
                  absorbance: torch.Tensor, 
                  noise_level: float = 0.01,
                  noise_type: str = 'gaussian') -> torch.Tensor:
        """
        Add noise to absorbance data for augmentation
        
        Args:
            absorbance: Input absorbance values
            noise_level: Noise intensity
            noise_type: Type of noise ('gaussian', 'uniform', 'multiplicative')
            
        Returns:
            Noisy absorbance data
        """
        if noise_type == 'gaussian':
            noise = torch.randn_like(absorbance) * noise_level
            return absorbance + noise
            
        elif noise_type == 'uniform':
            noise = (torch.rand_like(absorbance) - 0.5) * 2 * noise_level
            return absorbance + noise
            
        elif noise_type == 'multiplicative':
            noise = 1.0 + torch.randn_like(absorbance) * noise_level
            return absorbance * noise
            
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
            
    def smooth_spectrum(self, 
                       spectrum: torch.Tensor, 
                       kernel_size: int = 5,
                       sigma: float = 1.0) -> torch.Tensor:
        """
        Apply Gaussian smoothing to spectrum
        
        Args:
            spectrum: Input spectrum [batch_size] or [batch_size, n_wavelengths]
            kernel_size: Size of Gaussian kernel
            sigma: Standard deviation of Gaussian
            
        Returns:
            Smoothed spectrum
        """
        if spectrum.dim() == 1:
            spectrum = spectrum.unsqueeze(0).unsqueeze(0)  # [1, 1, n_wavelengths]
            squeeze_output = True
        elif spectrum.dim() == 2:
            spectrum = spectrum.unsqueeze(1)  # [batch_size, 1, n_wavelengths]
            squeeze_output = False
        else:
            squeeze_output = False
            
        # Create Gaussian kernel
        x = torch.arange(kernel_size, dtype=torch.float32, device=self.device)
        x = x - kernel_size // 2
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, -1)
        
        # Apply convolution
        padding = kernel_size // 2
        smoothed = F.conv1d(spectrum, kernel, padding=padding)
        
        if squeeze_output:
            return smoothed.squeeze()
        else:
            return smoothed.squeeze(1)
            
    def interpolate_spectrum(self,
                           spectrum: torch.Tensor,
                           wavelengths_old: torch.Tensor,
                           wavelengths_new: torch.Tensor) -> torch.Tensor:
        """
        Interpolate spectrum to new wavelength grid
        
        Args:
            spectrum: Input spectrum values
            wavelengths_old: Original wavelength grid
            wavelengths_new: Target wavelength grid
            
        Returns:
            Interpolated spectrum
        """
        # Use linear interpolation
        # Note: This is a simplified version, could use more sophisticated methods
        
        # Ensure wavelengths are sorted
        old_sorted, old_indices = torch.sort(wavelengths_old)
        spectrum_sorted = spectrum[old_indices] if spectrum.dim() == 1 else spectrum[:, old_indices]
        
        # Simple linear interpolation using searchsorted
        indices = torch.searchsorted(old_sorted, wavelengths_new, right=False)
        indices = torch.clamp(indices, 1, len(old_sorted) - 1)
        
        # Get neighboring points
        left_indices = indices - 1
        right_indices = indices
        
        left_wavelengths = old_sorted[left_indices]
        right_wavelengths = old_sorted[right_indices]
        left_values = spectrum_sorted[left_indices] if spectrum.dim() == 1 else spectrum_sorted[:, left_indices]
        right_values = spectrum_sorted[right_indices] if spectrum.dim() == 1 else spectrum_sorted[:, right_indices]
        
        # Linear interpolation
        weights = (wavelengths_new - left_wavelengths) / (right_wavelengths - left_wavelengths)
        interpolated = left_values + weights * (right_values - left_values)
        
        return interpolated
        
    def augment_batch(self,
                     batch: Dict[str, torch.Tensor],
                     augmentation_config: Dict[str, any]) -> Dict[str, torch.Tensor]:
        """
        Apply data augmentation to batch
        
        Args:
            batch: Input batch dictionary
            augmentation_config: Configuration for augmentation
            
        Returns:
            Augmented batch
        """
        augmented_batch = batch.copy()
        
        # Noise augmentation on absorbance
        if 'noise_level' in augmentation_config and 'target_absorbance' in batch:
            noise_level = augmentation_config['noise_level']
            noise_type = augmentation_config.get('noise_type', 'gaussian')
            
            augmented_batch['target_absorbance'] = self.add_noise(
                batch['target_absorbance'], noise_level, noise_type
            )
            
        # Wavelength shift (small random shifts)
        if 'wavelength_shift' in augmentation_config and 'wavelength' in batch:
            max_shift = augmentation_config['wavelength_shift']  # in normalized units
            shift = torch.rand(batch['wavelength'].shape[0], device=self.device) * 2 * max_shift - max_shift
            augmented_batch['wavelength'] = batch['wavelength'] + shift
            
        # Concentration noise (small random perturbations)
        if 'concentration_noise' in augmentation_config:
            noise_level = augmentation_config['concentration_noise']
            
            if 'c_source' in batch:
                noise = torch.randn_like(batch['c_source']) * noise_level
                augmented_batch['c_source'] = batch['c_source'] + noise
                
            if 'c_target' in batch:
                noise = torch.randn_like(batch['c_target']) * noise_level
                augmented_batch['c_target'] = batch['c_target'] + noise
                
        return augmented_batch
        
    def create_validation_grid(self,
                              c_range: Tuple[float, float] = (0, 60),
                              lambda_range: Tuple[float, float] = (200, 800),
                              n_concentrations: int = 21,
                              n_wavelengths: int = 101) -> Dict[str, torch.Tensor]:
        """
        Create dense grid for validation/visualization
        
        Args:
            c_range: Concentration range (ppb)
            lambda_range: Wavelength range (nm)
            n_concentrations: Number of concentration points
            n_wavelengths: Number of wavelength points
            
        Returns:
            Dictionary with grid data
        """
        # Create grids
        concentrations = torch.linspace(c_range[0], c_range[1], n_concentrations, device=self.device)
        wavelengths = torch.linspace(lambda_range[0], lambda_range[1], n_wavelengths, device=self.device)
        
        # Create meshgrid
        c_mesh, lambda_mesh = torch.meshgrid(concentrations, wavelengths, indexing='ij')
        
        # Flatten for processing
        c_flat = c_mesh.flatten()
        lambda_flat = lambda_mesh.flatten()
        
        # Normalize
        c_norm = self.normalize_concentrations(c_flat)
        lambda_norm = self.normalize_wavelengths(lambda_flat)
        
        return {
            'concentrations_raw': c_flat,
            'wavelengths_raw': lambda_flat,
            'concentrations_normalized': c_norm,
            'wavelengths_normalized': lambda_norm,
            'grid_shape': (n_concentrations, n_wavelengths)
        }
        
    def compute_normalization_params(self, dataset: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute normalization parameters from dataset
        
        Args:
            dataset: Dictionary with raw dataset tensors
            
        Returns:
            Dictionary with normalization parameters
        """
        params = {}
        
        # Concentration parameters
        if 'c_sources' in dataset and 'c_targets' in dataset:
            all_concentrations = torch.cat([dataset['c_sources'], dataset['c_targets']])
            params['c_mean'] = float(all_concentrations.mean())
            params['c_std'] = float(all_concentrations.std())
            
        # Wavelength parameters
        if 'wavelengths' in dataset:
            params['lambda_mean'] = float(dataset['wavelengths'].mean())
            params['lambda_std'] = float(dataset['wavelengths'].std())
            
        # Absorbance parameters
        if 'target_absorbances' in dataset:
            params['A_mean'] = float(dataset['target_absorbances'].mean())
            params['A_std'] = float(dataset['target_absorbances'].std())
            
        # Update stored parameters
        self.normalization_params.update(params)
        
        return params
        
    def get_preprocessing_stats(self) -> Dict[str, any]:
        """Get preprocessing configuration and statistics"""
        return {
            'device': str(self.device),
            'normalization_params': self.normalization_params,
            'default_params': self.default_params
        }