"""
GPU-optimized data loading for A100 training
Handles leave-one-out validation splits
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List, Optional


class SpectralDataset(Dataset):
    """Dataset for spectral concentration transitions"""
    
    def __init__(self,
                 concentration_values: List[float] = None,
                 wavelengths: Optional[np.ndarray] = None,
                 absorbance_data: Optional[np.ndarray] = None,
                 excluded_concentration_idx: Optional[int] = None,
                 normalize: bool = True,
                 device: torch.device = torch.device('cuda'),
                 csv_path: Optional[str] = None):
        """
        Initialize spectral dataset
        
        Args:
            concentration_values: List of concentration values in ppb
            wavelengths: Wavelength array (nm)
            absorbance_data: Absorbance matrix [n_concentrations, n_wavelengths]
            excluded_concentration_idx: Which concentration to exclude (for leave-one-out)
            normalize: Whether to normalize inputs
            device: Device for tensor storage
        """
        # Load from CSV if path provided
        if csv_path is not None:
            import pandas as pd
            df = pd.read_csv(csv_path)
            wavelengths = df['Wavelength'].values
            concentration_values = [float(col) for col in df.columns[1:]]
            absorbance_data = df.iloc[:, 1:].values.T  # Transpose to [n_conc, n_wave]
            
        # Use defaults if not provided
        if concentration_values is None:
            concentration_values = [0, 10, 20, 30, 40, 60]  # ppb
            
        if wavelengths is None:
            wavelengths = np.linspace(200, 800, 601)  # nm
            
        if absorbance_data is None:
            # Generate synthetic data only as fallback
            absorbance_data = self._generate_synthetic_data(
                concentration_values, wavelengths
            )
            
        self.concentration_values = np.array(concentration_values)
        self.wavelengths = wavelengths
        self.absorbance_data = absorbance_data
        self.excluded_idx = excluded_concentration_idx
        self.normalize = normalize
        self.device = device
        
        # Normalization parameters
        self.c_mean = 30.0  # ppb
        self.c_std = 30.0
        self.lambda_mean = 500.0  # nm
        self.lambda_std = 300.0
        self.A_mean = absorbance_data.mean()
        self.A_std = absorbance_data.std()
        
        # Create concentration transition pairs
        self._create_transition_pairs()
        
        # Move data to GPU
        self._prepare_gpu_tensors()
        
    def _generate_synthetic_data(self, concentrations: List[float],
                                wavelengths: np.ndarray) -> np.ndarray:
        """Generate synthetic absorbance data for testing"""
        n_conc = len(concentrations)
        n_wave = len(wavelengths)
        
        # Simple synthetic model: A = c * f(λ) with some nonlinearity
        absorbance = np.zeros((n_conc, n_wave))
        
        for i, c in enumerate(concentrations):
            for j, λ in enumerate(wavelengths):
                # Nonlinear response with wavelength-dependent sensitivity
                sensitivity = np.exp(-(λ - 450)**2 / (2 * 100**2))  # Peak at 450nm
                absorbance[i, j] = (c / 60) * sensitivity * (1 + 0.1 * np.sin(λ / 50))
                
        return absorbance
        
    def _create_transition_pairs(self):
        """Create all concentration transition pairs"""
        n_conc = len(self.concentration_values)
        pairs = []
        
        # Create all (source, target) pairs excluding self-transitions
        for i in range(n_conc):
            if self.excluded_idx is not None and i == self.excluded_idx:
                continue  # Skip excluded concentration
                
            for j in range(n_conc):
                if self.excluded_idx is not None and j == self.excluded_idx:
                    continue  # Skip excluded concentration
                    
                if i != j:  # No self-transitions
                    pairs.append((i, j))
                    
        self.transition_pairs = pairs
        
    def _prepare_gpu_tensors(self):
        """Pre-allocate and transfer data to GPU"""
        # Convert to tensors
        self.c_tensor = torch.tensor(
            self.concentration_values, dtype=torch.float32, device=self.device
        )
        self.lambda_tensor = torch.tensor(
            self.wavelengths, dtype=torch.float32, device=self.device
        )
        self.A_tensor = torch.tensor(
            self.absorbance_data, dtype=torch.float32, device=self.device
        )
        
        # Normalized versions
        if self.normalize:
            self.c_norm = (self.c_tensor - self.c_mean) / self.c_std
            self.lambda_norm = (self.lambda_tensor - self.lambda_mean) / self.lambda_std
            self.A_norm = (self.A_tensor - self.A_mean) / self.A_std
        else:
            self.c_norm = self.c_tensor
            self.lambda_norm = self.lambda_tensor
            self.A_norm = self.A_tensor
            
    def __len__(self) -> int:
        """Total number of training samples"""
        return len(self.transition_pairs) * len(self.wavelengths)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Get single training sample"""
        # Decompose index into pair and wavelength
        n_wavelengths = len(self.wavelengths)
        pair_idx = idx // n_wavelengths
        wave_idx = idx % n_wavelengths
        
        # Get concentration pair
        c_source_idx, c_target_idx = self.transition_pairs[pair_idx]
        
        # Get normalized values
        c_source = self.c_norm[c_source_idx]
        c_target = self.c_norm[c_target_idx]
        wavelength = self.lambda_norm[wave_idx]
        absorbance_target = self.A_norm[c_target_idx, wave_idx]
        
        return c_source, c_target, wavelength, absorbance_target
        
    def get_batch(self, batch_size: Optional[int] = None) -> Tuple[torch.Tensor, ...]:
        """Get entire dataset or batch as tensors (for parallel processing)"""
        if batch_size is None:
            # Return entire dataset
            n_pairs = len(self.transition_pairs)
            n_waves = len(self.wavelengths)
            
            # Pre-allocate tensors
            c_sources = torch.zeros(n_pairs * n_waves, device=self.device)
            c_targets = torch.zeros(n_pairs * n_waves, device=self.device)
            wavelengths = torch.zeros(n_pairs * n_waves, device=self.device)
            absorbances = torch.zeros(n_pairs * n_waves, device=self.device)
            
            idx = 0
            for pair_idx, (c_s_idx, c_t_idx) in enumerate(self.transition_pairs):
                for wave_idx in range(n_waves):
                    c_sources[idx] = self.c_norm[c_s_idx]
                    c_targets[idx] = self.c_norm[c_t_idx]
                    wavelengths[idx] = self.lambda_norm[wave_idx]
                    absorbances[idx] = self.A_norm[c_t_idx, wave_idx]
                    idx += 1
                    
            return c_sources, c_targets, wavelengths, absorbances
            
        else:
            # Random batch
            indices = torch.randint(0, len(self), (batch_size,), device=self.device)
            batch_data = [self[idx] for idx in indices]
            return tuple(torch.stack(tensors) for tensors in zip(*batch_data))
            
    def denormalize_absorbance(self, A_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalized absorbance back to original scale"""
        return A_norm * self.A_std + self.A_mean
        

def create_data_loaders(concentration_values: List[float] = None,
                       wavelengths: Optional[np.ndarray] = None,
                       absorbance_data: Optional[np.ndarray] = None,
                       batch_size: int = 2048,
                       device: torch.device = torch.device('cuda'),
                       csv_path: Optional[str] = None) -> dict:
    """
    Create data loaders for leave-one-out validation
    
    Args:
        concentration_values: List of concentration values
        wavelengths: Wavelength array
        absorbance_data: Absorbance matrix
        batch_size: Batch size for training
        device: Computation device
        
    Returns:
        Dictionary with datasets for each leave-one-out split
    """
    loaders = {}
    
    n_concentrations = 6 if concentration_values is None else len(concentration_values)
    
    for excluded_idx in range(n_concentrations):
        dataset = SpectralDataset(
            concentration_values=concentration_values,
            wavelengths=wavelengths,
            absorbance_data=absorbance_data,
            excluded_concentration_idx=excluded_idx,
            normalize=True,
            device=device,
            csv_path=csv_path
        )
        
        # For A100, we can often load entire dataset at once
        # So we'll use a simple DataLoader with large batch size
        loader = DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=True,
            pin_memory=False,  # Data already on GPU
            num_workers=0  # GPU-resident data
        )
        
        loaders[f'exclude_{excluded_idx}'] = {
            'dataset': dataset,
            'loader': loader,
            'excluded_concentration': dataset.concentration_values[excluded_idx]
        }
        
    return loaders