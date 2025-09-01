#!/usr/bin/env python3
"""
Data loader for UV-Vis spectral data
Loads CSV and creates training pairs for geodesic interpolation
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List


class SpectralDataset(Dataset):
    """Dataset for spectral interpolation with concentration pairs"""
    
    def __init__(self, csv_path: str = "data/0.30MB_AuNP_As.csv", 
                 exclude_concentration_idx: int = None):
        """
        Load spectral data and create concentration transition pairs
        
        Args:
            csv_path: Path to CSV file
            exclude_concentration_idx: Index of concentration to exclude (for validation)
        """
        # Load data
        self.df = pd.read_csv(csv_path)
        self.wavelengths = self.df['Wavelength'].values  # 601 wavelengths
        self.concentration_columns = [col for col in self.df.columns[1:]]
        self.concentrations = np.array([float(col) for col in self.concentration_columns])
        self.absorbance_matrix = self.df.iloc[:, 1:].values  # 601 x 6
        
        # Store normalization parameters
        self.wavelength_mean = 500.0
        self.wavelength_std = 300.0
        self.concentration_mean = 30.0
        self.concentration_std = 30.0
        self.absorbance_mean = self.absorbance_matrix.mean()
        self.absorbance_std = self.absorbance_matrix.std()
        
        # Create training pairs (excluding holdout if specified)
        self.pairs = []
        concentration_indices = list(range(len(self.concentrations)))
        
        if exclude_concentration_idx is not None:
            concentration_indices.remove(exclude_concentration_idx)
        
        # Generate all concentration transition pairs
        for i, c_idx_source in enumerate(concentration_indices):
            for j, c_idx_target in enumerate(concentration_indices):
                if c_idx_source != c_idx_target:
                    # For each wavelength, create a training sample
                    for wl_idx in range(len(self.wavelengths)):
                        self.pairs.append({
                            'c_source': self.concentrations[c_idx_source],
                            'c_target': self.concentrations[c_idx_target],
                            'wavelength': self.wavelengths[wl_idx],
                            'absorbance_target': self.absorbance_matrix[wl_idx, c_idx_target],
                            'c_source_idx': c_idx_source,
                            'c_target_idx': c_idx_target,
                            'wl_idx': wl_idx
                        })
    
    def normalize_wavelength(self, wavelength: float) -> float:
        """Normalize wavelength to [-1, 1]"""
        return (wavelength - self.wavelength_mean) / self.wavelength_std
    
    def normalize_concentration(self, concentration: float) -> float:
        """Normalize concentration to [-1, 1]"""
        return (concentration - self.concentration_mean) / self.concentration_std
    
    def normalize_absorbance(self, absorbance: float) -> float:
        """Normalize absorbance"""
        return (absorbance - self.absorbance_mean) / self.absorbance_std
    
    def denormalize_absorbance(self, absorbance_norm: float) -> float:
        """Denormalize absorbance"""
        return absorbance_norm * self.absorbance_std + self.absorbance_mean
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> dict:
        """Return normalized training sample"""
        pair = self.pairs[idx]
        
        return {
            'c_source': self.normalize_concentration(pair['c_source']),
            'c_target': self.normalize_concentration(pair['c_target']),
            'wavelength': self.normalize_wavelength(pair['wavelength']),
            'absorbance_target': self.normalize_absorbance(pair['absorbance_target']),
            # Keep raw values for analysis
            'c_source_raw': pair['c_source'],
            'c_target_raw': pair['c_target'],
            'wavelength_raw': pair['wavelength'],
            'absorbance_target_raw': pair['absorbance_target']
        }


def create_data_loaders(batch_size: int = 32, 
                       exclude_concentration_idx: int = None) -> Tuple[DataLoader, SpectralDataset]:
    """
    Create data loader for training
    
    Args:
        batch_size: Batch size for training
        exclude_concentration_idx: Concentration index to exclude for validation
    
    Returns:
        DataLoader and Dataset objects
    """
    dataset = SpectralDataset(exclude_concentration_idx=exclude_concentration_idx)
    
    # Custom collate function to handle dict batches
    def collate_fn(batch):
        return {
            'c_source': torch.tensor([item['c_source'] for item in batch], dtype=torch.float32),
            'c_target': torch.tensor([item['c_target'] for item in batch], dtype=torch.float32),
            'wavelength': torch.tensor([item['wavelength'] for item in batch], dtype=torch.float32),
            'absorbance_target': torch.tensor([item['absorbance_target'] for item in batch], dtype=torch.float32),
            'c_source_raw': torch.tensor([item['c_source_raw'] for item in batch], dtype=torch.float32),
            'c_target_raw': torch.tensor([item['c_target_raw'] for item in batch], dtype=torch.float32),
            'wavelength_raw': torch.tensor([item['wavelength_raw'] for item in batch], dtype=torch.float32),
            'absorbance_target_raw': torch.tensor([item['absorbance_target_raw'] for item in batch], dtype=torch.float32)
        }
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    return dataloader, dataset


def get_holdout_test_data(holdout_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get test data for a specific held-out concentration
    
    Args:
        holdout_idx: Index of concentration to use as test
    
    Returns:
        wavelengths, true_absorbance, concentration value
    """
    df = pd.read_csv("data/0.30MB_AuNP_As.csv")
    wavelengths = df['Wavelength'].values
    concentration_columns = [col for col in df.columns[1:]]
    concentrations = np.array([float(col) for col in concentration_columns])
    absorbance_matrix = df.iloc[:, 1:].values
    
    test_concentration = concentrations[holdout_idx]
    test_absorbance = absorbance_matrix[:, holdout_idx]
    
    return wavelengths, test_absorbance, test_concentration


if __name__ == "__main__":
    # Test data loading
    print("Testing data loader...")
    
    # Load full dataset
    dataset = SpectralDataset()
    print(f"Total training pairs: {len(dataset)}")
    print(f"Expected: 6 * 5 * 601 = {6 * 5 * 601}")
    
    # Test single sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"c_source normalized: {sample['c_source']:.3f} (raw: {sample['c_source_raw']})")
    print(f"c_target normalized: {sample['c_target']:.3f} (raw: {sample['c_target_raw']})")
    print(f"wavelength normalized: {sample['wavelength']:.3f} (raw: {sample['wavelength_raw']})")
    
    # Test with holdout
    dataset_holdout = SpectralDataset(exclude_concentration_idx=5)  # Exclude 60 ppb
    print(f"\nWith holdout (60 ppb): {len(dataset_holdout)} pairs")
    print(f"Expected: 5 * 4 * 601 = {5 * 4 * 601}")
    
    # Test DataLoader
    dataloader, _ = create_data_loaders(batch_size=32)
    batch = next(iter(dataloader))
    print(f"\nBatch shapes:")
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {val.shape}")