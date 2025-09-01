#!/usr/bin/env python3
"""
GPU-Resident Data Generation
Creates synthetic spectral data directly on MPS/CPU for zero-copy training
"""

import torch
import numpy as np
from typing import Tuple, Dict, Optional
import pandas as pd
from pathlib import Path


class SpectralDataGenerator:
    """Generate and manage spectral data entirely on device"""
    
    def __init__(self, device: torch.device, use_real_data: bool = True):
        """
        Initialize data generator
        
        Args:
            device: Target device (MPS/CPU)
            use_real_data: Whether to load real arsenic data
        """
        self.device = device
        self.use_real_data = use_real_data
        
        # Standard parameters
        self.wavelengths = torch.linspace(200, 800, 601, device=device)
        self.concentrations = torch.tensor([0, 10, 20, 30, 40, 60], dtype=torch.float32, device=device)
        
        # Normalization parameters
        self.c_min, self.c_max = 0, 60
        self.wl_min, self.wl_max = 200, 800
        
        # Load or generate data
        if use_real_data:
            self.spectra = self._load_real_data()
        else:
            self.spectra = self._generate_synthetic_data()
        
        # Pre-compute all transitions
        self.transitions = self._compute_all_transitions()
        
        # Cache normalized values
        self.wavelengths_norm = self.normalize_wavelengths(self.wavelengths)
        self.concentrations_norm = self.normalize_concentrations(self.concentrations)
    
    def _load_real_data(self) -> torch.Tensor:
        """Load real arsenic spectral data"""
        data_path = Path(__file__).parent.parent.parent / "data" / "0.30MB_AuNP_As.csv"
        
        if data_path.exists():
            df = pd.read_csv(data_path)
            spectra_np = df.iloc[:, 1:].values  # (601, 6)
            spectra = torch.tensor(spectra_np, dtype=torch.float32, device=self.device)
            return spectra.T  # (6, 601) for easier indexing
        else:
            print(f"Warning: Real data not found at {data_path}, using synthetic")
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> torch.Tensor:
        """Generate synthetic non-monotonic spectral data"""
        # Create non-monotonic spectral response
        n_conc = len(self.concentrations)
        n_wl = len(self.wavelengths)
        
        # Base spectrum with peak around 520nm (gold nanoparticle resonance)
        wl_centered = (self.wavelengths - 520) / 100
        base_spectrum = torch.exp(-0.5 * wl_centered**2)
        
        # Add concentration-dependent shift and broadening
        spectra = torch.zeros(n_conc, n_wl, device=self.device)
        
        for i, conc in enumerate(self.concentrations):
            # Non-linear, non-monotonic response
            shift = 0.1 * torch.sin(conc / 10)  # Non-monotonic shift
            broadening = 1 + 0.02 * conc  # Broadening with concentration
            intensity = 0.3 + 0.01 * conc - 0.0001 * conc**2  # Non-monotonic intensity
            
            wl_shifted = (self.wavelengths - 520 - shift * 100) / (100 * broadening)
            spectra[i] = intensity * torch.exp(-0.5 * wl_shifted**2)
            
            # Add some noise
            spectra[i] += 0.001 * torch.randn(n_wl, device=self.device)
        
        return spectra
    
    def _compute_all_transitions(self) -> Dict[str, torch.Tensor]:
        """Pre-compute all possible concentration transitions"""
        n_conc = len(self.concentrations)
        transitions = []
        
        for i in range(n_conc):
            for j in range(n_conc):
                if i != j:
                    transitions.append((i, j))
        
        # Convert to tensors
        transitions_tensor = torch.tensor(transitions, device=self.device)
        
        return {
            'pairs': transitions_tensor,
            'source_idx': transitions_tensor[:, 0],
            'target_idx': transitions_tensor[:, 1],
            'source_conc': self.concentrations[transitions_tensor[:, 0]],
            'target_conc': self.concentrations[transitions_tensor[:, 1]],
            'source_conc_norm': self.normalize_concentrations(
                self.concentrations[transitions_tensor[:, 0]]
            ),
            'target_conc_norm': self.normalize_concentrations(
                self.concentrations[transitions_tensor[:, 1]]
            ),
        }
    
    def normalize_concentrations(self, c: torch.Tensor) -> torch.Tensor:
        """Normalize concentrations to [-1, 1]"""
        return 2 * (c - self.c_min) / (self.c_max - self.c_min) - 1
    
    def normalize_wavelengths(self, wl: torch.Tensor) -> torch.Tensor:
        """Normalize wavelengths to [-1, 1]"""
        return 2 * (wl - self.wl_min) / (self.wl_max - self.wl_min) - 1
    
    def denormalize_concentrations(self, c_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize concentrations from [-1, 1]"""
        return (c_norm + 1) * (self.c_max - self.c_min) / 2 + self.c_min
    
    def denormalize_wavelengths(self, wl_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize wavelengths from [-1, 1]"""
        return (wl_norm + 1) * (self.wl_max - self.wl_min) / 2 + self.wl_min
    
    def get_batch(self, batch_size: int, 
                  wavelength_subset: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Get a batch of training data
        
        Args:
            batch_size: Number of samples
            wavelength_subset: Optional number of wavelengths to sample
        
        Returns:
            Dictionary with batch data
        """
        n_transitions = len(self.transitions['pairs'])
        n_wavelengths = len(self.wavelengths) if wavelength_subset is None else wavelength_subset
        
        # Sample transitions
        trans_idx = torch.randint(0, n_transitions, (batch_size,), device=self.device)
        
        # Sample wavelengths
        if wavelength_subset is None:
            wl_idx = torch.arange(len(self.wavelengths), device=self.device)
            wl_idx = wl_idx.repeat(batch_size, 1).flatten()
            trans_idx = trans_idx.repeat_interleave(len(self.wavelengths))
        else:
            wl_idx = torch.randint(0, len(self.wavelengths), 
                                 (batch_size * wavelength_subset,), device=self.device)
            trans_idx = trans_idx.repeat_interleave(wavelength_subset)
        
        # Get source and target concentrations
        source_idx = self.transitions['source_idx'][trans_idx]
        target_idx = self.transitions['target_idx'][trans_idx]
        
        # Get normalized values
        source_conc = self.transitions['source_conc_norm'][trans_idx]
        target_conc = self.transitions['target_conc_norm'][trans_idx]
        wavelengths = self.wavelengths_norm[wl_idx]
        
        # Get target absorbances
        target_spectra = self.spectra[target_idx, wl_idx]
        
        return {
            'source_conc': source_conc,
            'target_conc': target_conc,
            'wavelengths': wavelengths,
            'target_absorbance': target_spectra,
            'source_idx': source_idx,
            'target_idx': target_idx,
            'wavelength_idx': wl_idx,
            'batch_size': batch_size,
            'n_wavelengths': n_wavelengths,
        }
    
    def get_validation_set(self, n_samples: int = 100) -> Dict[str, torch.Tensor]:
        """Get validation set with fixed random sampling"""
        generator = torch.Generator(device=self.device)
        generator.manual_seed(42)  # Fixed seed for reproducibility
        
        n_transitions = len(self.transitions['pairs'])
        trans_idx = torch.randint(0, n_transitions, (n_samples,), 
                                 device=self.device, generator=generator)
        
        # Use all wavelengths for validation
        wl_idx = torch.arange(len(self.wavelengths), device=self.device)
        wl_idx = wl_idx.repeat(n_samples, 1).flatten()
        trans_idx = trans_idx.repeat_interleave(len(self.wavelengths))
        
        source_idx = self.transitions['source_idx'][trans_idx]
        target_idx = self.transitions['target_idx'][trans_idx]
        
        return {
            'source_conc': self.transitions['source_conc_norm'][trans_idx],
            'target_conc': self.transitions['target_conc_norm'][trans_idx],
            'wavelengths': self.wavelengths_norm[wl_idx],
            'target_absorbance': self.spectra[target_idx, wl_idx],
            'source_idx': source_idx,
            'target_idx': target_idx,
            'wavelength_idx': wl_idx,
        }
    
    def get_interpolation_test(self, test_conc: float) -> Dict[str, torch.Tensor]:
        """
        Get test data for interpolation at unseen concentration
        
        Args:
            test_conc: Test concentration in ppb
        
        Returns:
            Test data dictionary
        """
        # Find nearest training concentrations
        conc_np = self.concentrations.cpu().numpy()
        idx_below = np.where(conc_np < test_conc)[0]
        idx_above = np.where(conc_np > test_conc)[0]
        
        if len(idx_below) == 0 or len(idx_above) == 0:
            raise ValueError(f"Test concentration {test_conc} outside training range")
        
        nearest_below = idx_below[-1]
        nearest_above = idx_above[0]
        
        # Normalize test concentration
        test_conc_tensor = torch.tensor([test_conc], device=self.device)
        test_conc_norm = self.normalize_concentrations(test_conc_tensor)
        
        # Create test transitions
        source_conc = self.concentrations_norm[nearest_below].unsqueeze(0)
        target_conc = test_conc_norm
        
        # Use all wavelengths
        wavelengths = self.wavelengths_norm
        
        # Estimate target spectrum (linear interpolation for reference)
        alpha = (test_conc - conc_np[nearest_below]) / \
                (conc_np[nearest_above] - conc_np[nearest_below])
        estimated_spectrum = (1 - alpha) * self.spectra[nearest_below] + \
                           alpha * self.spectra[nearest_above]
        
        return {
            'source_conc': source_conc.repeat(len(wavelengths)),
            'target_conc': target_conc.repeat(len(wavelengths)),
            'wavelengths': wavelengths,
            'estimated_absorbance': estimated_spectrum,
            'test_concentration_ppb': test_conc,
            'alpha': alpha,
        }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Calculate memory usage of cached data"""
        total_elements = 0
        
        # Spectra
        total_elements += self.spectra.numel()
        
        # Transitions
        for key, tensor in self.transitions.items():
            if isinstance(tensor, torch.Tensor):
                total_elements += tensor.numel()
        
        # Normalized values
        total_elements += self.wavelengths_norm.numel()
        total_elements += self.concentrations_norm.numel()
        
        # Calculate memory (assuming float32)
        memory_mb = total_elements * 4 / (1024 * 1024)
        
        return {
            'total_elements': total_elements,
            'memory_mb': memory_mb,
            'spectra_shape': list(self.spectra.shape),
            'n_transitions': len(self.transitions['pairs']),
        }


if __name__ == "__main__":
    # Test data generator
    print("Testing Spectral Data Generator...")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Initialize generator
    generator = SpectralDataGenerator(device, use_real_data=True)
    print(f"\nData loaded successfully")
    print(f"Spectra shape: {generator.spectra.shape}")
    print(f"Number of transitions: {len(generator.transitions['pairs'])}")
    
    # Test batch generation
    batch = generator.get_batch(batch_size=32, wavelength_subset=100)
    print(f"\nBatch generated:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape {value.shape}, device {value.device}")
        else:
            print(f"  {key}: {value}")
    
    # Test validation set
    val_set = generator.get_validation_set(n_samples=10)
    print(f"\nValidation set:")
    print(f"  Total samples: {len(val_set['source_conc'])}")
    
    # Test interpolation
    try:
        test_data = generator.get_interpolation_test(test_conc=25)
        print(f"\nInterpolation test at 25 ppb:")
        print(f"  Alpha: {test_data['alpha']:.3f}")
        print(f"  Estimated spectrum shape: {test_data['estimated_absorbance'].shape}")
    except ValueError as e:
        print(f"Interpolation test failed: {e}")
    
    # Check memory usage
    memory = generator.get_memory_usage()
    print(f"\nMemory usage:")
    for key, value in memory.items():
        print(f"  {key}: {value}")