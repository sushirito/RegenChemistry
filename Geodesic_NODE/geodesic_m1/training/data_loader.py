"""
GPU-optimized data pipeline for M1 Mac MPS
Handles spectral concentration transition data
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Optional, Iterator
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class SpectralDataset(Dataset):
    """Dataset for spectral concentration transitions optimized for M1 Mac"""
    
    def __init__(self,
                 concentration_values: List[float] = None,
                 wavelengths: np.ndarray = None,
                 absorbance_data: np.ndarray = None,
                 excluded_concentration_idx: Optional[int] = None,
                 normalize: bool = True,
                 device: torch.device = torch.device('mps')):
        """
        Initialize spectral dataset
        
        Args:
            concentration_values: List of concentration values [0, 10, 20, 30, 40, 60] ppb
            wavelengths: Array of wavelength values [601]
            absorbance_data: Absorbance matrix [6, 601] (concentrations × wavelengths)
            excluded_concentration_idx: Which concentration to exclude (for leave-one-out)
            normalize: Whether to normalize inputs
            device: Target device (MPS for M1)
        """
        if concentration_values is None:
            concentration_values = [0, 10, 20, 30, 40, 60]  # ppb
            
        if wavelengths is None:
            wavelengths = np.linspace(200, 800, 601, dtype=np.float32)  # nm
            
        self.concentration_values = np.array(concentration_values, dtype=np.float32)
        self.wavelengths = np.array(wavelengths, dtype=np.float32)
        self.absorbance_data = absorbance_data.astype(np.float32) if absorbance_data is not None else None
        self.excluded_concentration_idx = excluded_concentration_idx
        self.normalize = normalize
        self.device = device
        
        # Create normalization parameters
        if normalize:
            self.c_mean, self.c_std = 30.0, 30.0  # Normalize to [-1, 1]
            self.lambda_mean, self.lambda_std = 500.0, 300.0  # Normalize to [-1, 1]
            
            if absorbance_data is not None:
                self.A_mean = float(absorbance_data.mean())
                self.A_std = float(absorbance_data.std())
            else:
                self.A_mean, self.A_std = 0.0, 1.0
                
        # Generate training pairs (all concentration transitions)
        self._generate_training_pairs()
        
    def _generate_training_pairs(self):
        """Generate all possible concentration transition pairs"""
        n_concentrations = len(self.concentration_values)
        n_wavelengths = len(self.wavelengths)
        
        # Create all pairs excluding self-transitions
        pairs = []
        for i in range(n_concentrations):
            for j in range(n_concentrations):
                if i != j:  # Exclude self-transitions
                    # Skip if this concentration is excluded for leave-one-out
                    if (self.excluded_concentration_idx is not None and 
                        (i == self.excluded_concentration_idx or j == self.excluded_concentration_idx)):
                        continue
                        
                    for k in range(n_wavelengths):
                        pairs.append((i, j, k))  # (source_idx, target_idx, wavelength_idx)
                        
        self.training_pairs = pairs
        print(f"Generated {len(pairs)} training pairs (excluded concentration {self.excluded_concentration_idx})")
        
    def __len__(self) -> int:
        return len(self.training_pairs)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example"""
        source_idx, target_idx, wavelength_idx = self.training_pairs[idx]
        
        # Get values
        c_source = self.concentration_values[source_idx]
        c_target = self.concentration_values[target_idx]
        wavelength = self.wavelengths[wavelength_idx]
        
        # Get target absorbance if available
        if self.absorbance_data is not None:
            target_absorbance = self.absorbance_data[target_idx, wavelength_idx]
        else:
            target_absorbance = 0.0  # Placeholder
            
        # Normalize inputs
        if self.normalize:
            c_source_norm = (c_source - self.c_mean) / self.c_std
            c_target_norm = (c_target - self.c_mean) / self.c_std
            wavelength_norm = (wavelength - self.lambda_mean) / self.lambda_std
            target_absorbance_norm = (target_absorbance - self.A_mean) / self.A_std
        else:
            c_source_norm = c_source
            c_target_norm = c_target
            wavelength_norm = wavelength
            target_absorbance_norm = target_absorbance
            
        # Return as tensors on CPU (will be moved to device by DataLoader)
        return {
            'c_source': torch.tensor(c_source_norm, dtype=torch.float32),
            'c_target': torch.tensor(c_target_norm, dtype=torch.float32),
            'wavelength': torch.tensor(wavelength_norm, dtype=torch.float32),
            'target_absorbance': torch.tensor(target_absorbance_norm, dtype=torch.float32),
            'source_idx': torch.tensor(source_idx, dtype=torch.long),
            'target_idx': torch.tensor(target_idx, dtype=torch.long),
            'wavelength_idx': torch.tensor(wavelength_idx, dtype=torch.long)
        }
        
    def get_validation_data(self, excluded_concentration_idx: int) -> Dict[str, torch.Tensor]:
        """Get validation data for specific excluded concentration"""
        n_wavelengths = len(self.wavelengths)
        
        # Create data for interpolating TO the excluded concentration
        validation_data = []
        for source_idx in range(len(self.concentration_values)):
            if source_idx == excluded_concentration_idx:
                continue
                
            for wavelength_idx in range(n_wavelengths):
                c_source = self.concentration_values[source_idx]
                c_target = self.concentration_values[excluded_concentration_idx]
                wavelength = self.wavelengths[wavelength_idx]
                
                if self.absorbance_data is not None:
                    target_absorbance = self.absorbance_data[excluded_concentration_idx, wavelength_idx]
                else:
                    target_absorbance = 0.0
                    
                # Normalize
                if self.normalize:
                    c_source_norm = (c_source - self.c_mean) / self.c_std
                    c_target_norm = (c_target - self.c_mean) / self.c_std
                    wavelength_norm = (wavelength - self.lambda_mean) / self.lambda_std
                    target_absorbance_norm = (target_absorbance - self.A_mean) / self.A_std
                else:
                    c_source_norm = c_source
                    c_target_norm = c_target
                    wavelength_norm = wavelength
                    target_absorbance_norm = target_absorbance
                    
                validation_data.append({
                    'c_source': c_source_norm,
                    'c_target': c_target_norm,
                    'wavelength': wavelength_norm,
                    'target_absorbance': target_absorbance_norm
                })
                
        # Stack into tensors
        if validation_data:
            return {
                'c_sources': torch.tensor([d['c_source'] for d in validation_data], 
                                        dtype=torch.float32, device=self.device),
                'c_targets': torch.tensor([d['c_target'] for d in validation_data], 
                                        dtype=torch.float32, device=self.device),
                'wavelengths': torch.tensor([d['wavelength'] for d in validation_data], 
                                          dtype=torch.float32, device=self.device),
                'target_absorbances': torch.tensor([d['target_absorbance'] for d in validation_data], 
                                                 dtype=torch.float32, device=self.device)
            }
        else:
            return {}


class SpectralDataLoader:
    """Data loader optimized for M1 Mac MPS training"""
    
    def __init__(self,
                 dataset: SpectralDataset,
                 batch_size: int = 1024,
                 shuffle: bool = True,
                 num_workers: int = 0,  # MPS works better with 0 workers
                 pin_memory: bool = False):  # Not needed for unified memory
        """
        Initialize M1-optimized data loader
        
        Args:
            dataset: Spectral dataset
            batch_size: Batch size (optimized for M1)
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes (0 for MPS)
            pin_memory: Pin memory (disabled for unified memory)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Create PyTorch DataLoader with M1 optimizations
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,  # Consistent batch sizes for MPS
            persistent_workers=False
        )
        
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over batches"""
        for batch in self.dataloader:
            # Batch is already on device from dataset
            yield batch
            
    def __len__(self) -> int:
        return len(self.dataloader)
        
    def get_batch_info(self) -> Dict[str, int]:
        """Get information about batching"""
        return {
            'dataset_size': len(self.dataset),
            'batch_size': self.batch_size,
            'num_batches': len(self.dataloader),
            'samples_per_epoch': len(self.dataloader) * self.batch_size
        }


def load_spectral_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load spectral data from CSV file
    
    Args:
        csv_path: Path to CSV file with spectral data
        
    Returns:
        Tuple of (wavelengths, absorbance_matrix)
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Extract wavelengths (assuming first column or row contains wavelengths)
        if 'wavelength' in df.columns:
            wavelengths = df['wavelength'].values
            absorbance_data = df.drop('wavelength', axis=1).values.T
        else:
            # Assume wavelengths are in first row or first column
            wavelengths = df.iloc[0, 1:].values
            absorbance_data = df.iloc[1:, 1:].values
            
        return wavelengths.astype(np.float32), absorbance_data.astype(np.float32)
        
    except Exception as e:
        print(f"Error loading spectral data: {e}")
        # Return default synthetic data
        wavelengths = np.linspace(200, 800, 601)
        absorbance_data = np.random.rand(6, 601) * 2.0  # Placeholder
        return wavelengths.astype(np.float32), absorbance_data.astype(np.float32)


def create_leave_one_out_datasets(wavelengths: np.ndarray,
                                 absorbance_data: np.ndarray,
                                 device: torch.device = torch.device('mps')) -> List[SpectralDataset]:
    """
    Create all 6 leave-one-out datasets
    
    Args:
        wavelengths: Wavelength array [601]
        absorbance_data: Absorbance matrix [6, 601]
        device: Target device
        
    Returns:
        List of 6 datasets, each excluding one concentration
    """
    datasets = []
    concentration_values = [0, 10, 20, 30, 40, 60]  # ppb
    
    for excluded_idx in range(6):
        dataset = SpectralDataset(
            concentration_values=concentration_values,
            wavelengths=wavelengths,
            absorbance_data=absorbance_data,
            excluded_concentration_idx=excluded_idx,
            normalize=True,
            device=device
        )
        datasets.append(dataset)
        
    return datasets