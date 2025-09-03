"""
GPU-resident data generation for M1 Mac
Generates synthetic and processes real spectral data
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, List
import pandas as pd
from pathlib import Path


class SpectralDataGenerator:
    """Generates spectral data directly on M1 Mac GPU memory"""
    
    def __init__(self,
                 device: torch.device = torch.device('mps'),
                 seed: Optional[int] = 42):
        """
        Initialize spectral data generator
        
        Args:
            device: Target device (MPS for M1)
            seed: Random seed for reproducibility
        """
        self.device = device
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        # Default spectral parameters
        self.concentration_values = torch.tensor([0, 10, 20, 30, 40, 60], 
                                               dtype=torch.float32, device=device)
        self.wavelength_range = (200, 800)  # nm
        self.n_wavelengths = 601
        
        # Generate wavelength grid
        self.wavelengths = torch.linspace(
            self.wavelength_range[0], 
            self.wavelength_range[1], 
            self.n_wavelengths,
            device=device,
            dtype=torch.float32
        )
        
    def generate_synthetic_data(self,
                              noise_level: float = 0.01,
                              peak_wavelengths: List[float] = None) -> torch.Tensor:
        """
        Generate synthetic spectral absorbance data
        
        Args:
            noise_level: Gaussian noise standard deviation
            peak_wavelengths: Primary absorption peaks (nm)
            
        Returns:
            Absorbance matrix [6, 601] (concentrations × wavelengths)
        """
        if peak_wavelengths is None:
            peak_wavelengths = [520, 650]  # Default peaks for gold nanoparticles
            
        n_concentrations = len(self.concentration_values)
        absorbance_data = torch.zeros(n_concentrations, self.n_wavelengths, 
                                    device=self.device, dtype=torch.float32)
        
        for i, concentration in enumerate(self.concentration_values):
            # Concentration-dependent spectral response
            conc_normalized = concentration / 60.0  # Normalize to [0, 1]
            
            spectrum = torch.zeros_like(self.wavelengths)
            
            # Add absorption peaks
            for peak_wl in peak_wavelengths:
                # Non-monotonic response: gray → blue → gray
                if conc_normalized < 0.5:
                    # Lower concentrations: increasing absorption
                    peak_intensity = conc_normalized * 2.0
                else:
                    # Higher concentrations: decreasing due to aggregation
                    peak_intensity = 2.0 * (1.0 - conc_normalized)
                    
                # Gaussian peak
                peak_width = 50 + 20 * conc_normalized  # Broadening with concentration
                gaussian = peak_intensity * torch.exp(
                    -((self.wavelengths - peak_wl) / peak_width) ** 2
                )
                spectrum += gaussian
                
            # Add baseline absorption
            baseline = 0.1 * conc_normalized
            spectrum += baseline
            
            # Add realistic noise
            if noise_level > 0:
                noise = torch.randn_like(spectrum) * noise_level
                spectrum += noise
                
            # Ensure non-negative
            spectrum = torch.clamp(spectrum, min=0.0)
            
            absorbance_data[i] = spectrum
            
        return absorbance_data
        
    def load_real_data(self, csv_path: str) -> torch.Tensor:
        """
        Load real spectral data from CSV file
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Absorbance matrix [6, 601] on device
        """
        try:
            # Load CSV data
            df = pd.read_csv(csv_path)
            
            # Extract absorbance data
            if 'wavelength' in df.columns:
                # Wavelengths in first column, concentrations in others
                wavelengths_csv = df['wavelength'].values
                absorbance_csv = df.drop('wavelength', axis=1).values.T
            else:
                # Assume first row is wavelengths, rest is data
                wavelengths_csv = df.iloc[0, 1:].values
                absorbance_csv = df.iloc[1:, 1:].values
                
            # Interpolate to standard grid if needed
            if len(wavelengths_csv) != self.n_wavelengths:
                print(f"⚠️  Interpolating from {len(wavelengths_csv)} to {self.n_wavelengths} wavelengths")
                absorbance_interpolated = self._interpolate_spectra(
                    wavelengths_csv, absorbance_csv, self.wavelengths.cpu().numpy()
                )
            else:
                absorbance_interpolated = absorbance_csv
                
            # Convert to tensor on device
            absorbance_data = torch.tensor(
                absorbance_interpolated, 
                dtype=torch.float32, 
                device=self.device
            )
            
            print(f"✅ Loaded real spectral data: {absorbance_data.shape}")
            return absorbance_data
            
        except Exception as e:
            print(f"❌ Error loading real data: {e}")
            print("   Falling back to synthetic data...")
            return self.generate_synthetic_data()
            
    def _interpolate_spectra(self, 
                           wavelengths_old: np.ndarray, 
                           absorbance_old: np.ndarray,
                           wavelengths_new: np.ndarray) -> np.ndarray:
        """Interpolate spectral data to new wavelength grid"""
        from scipy.interpolate import interp1d
        
        n_concentrations = absorbance_old.shape[0]
        absorbance_new = np.zeros((n_concentrations, len(wavelengths_new)))
        
        for i in range(n_concentrations):
            interpolator = interp1d(
                wavelengths_old, absorbance_old[i],
                kind='cubic', bounds_error=False, fill_value='extrapolate'
            )
            absorbance_new[i] = interpolator(wavelengths_new)
            
        return absorbance_new
        
    def create_training_pairs(self, 
                            absorbance_data: torch.Tensor,
                            excluded_concentration_idx: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Create all training pairs for concentration transitions
        
        Args:
            absorbance_data: Absorbance matrix [6, 601]
            excluded_concentration_idx: Concentration to exclude (leave-one-out)
            
        Returns:
            Dictionary with training data tensors
        """
        n_concentrations = absorbance_data.shape[0]
        n_wavelengths = absorbance_data.shape[1]
        
        # Generate all transition pairs
        c_sources = []
        c_targets = []
        wavelengths_expanded = []
        target_absorbances = []
        
        for i in range(n_concentrations):
            for j in range(n_concentrations):
                if i == j:  # Skip self-transitions
                    continue
                    
                # Skip if concentration is excluded
                if excluded_concentration_idx is not None:
                    if i == excluded_concentration_idx or j == excluded_concentration_idx:
                        continue
                        
                # Add this transition for all wavelengths
                for k in range(n_wavelengths):
                    c_sources.append(self.concentration_values[i])
                    c_targets.append(self.concentration_values[j])
                    wavelengths_expanded.append(self.wavelengths[k])
                    target_absorbances.append(absorbance_data[j, k])
                    
        # Convert to tensors
        training_data = {
            'c_sources': torch.stack(c_sources),
            'c_targets': torch.stack(c_targets), 
            'wavelengths': torch.stack(wavelengths_expanded),
            'target_absorbances': torch.stack(target_absorbances)
        }
        
        print(f"📊 Created {len(c_sources)} training pairs "
              f"(excluded concentration {excluded_concentration_idx})")
              
        return training_data
        
    def normalize_data(self, data: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Normalize training data for better convergence
        
        Args:
            data: Dictionary with raw data tensors
            
        Returns:
            Tuple of (normalized_data, normalization_params)
        """
        # Compute normalization parameters
        c_mean, c_std = 30.0, 30.0  # Concentration normalization
        lambda_mean, lambda_std = 500.0, 300.0  # Wavelength normalization
        A_mean = data['target_absorbances'].mean()
        A_std = data['target_absorbances'].std()
        
        # Normalize data
        normalized_data = {
            'c_sources': (data['c_sources'] - c_mean) / c_std,
            'c_targets': (data['c_targets'] - c_mean) / c_std,
            'wavelengths': (data['wavelengths'] - lambda_mean) / lambda_std,
            'target_absorbances': (data['target_absorbances'] - A_mean) / A_std
        }
        
        normalization_params = {
            'c_mean': c_mean,
            'c_std': c_std,
            'lambda_mean': lambda_mean,
            'lambda_std': lambda_std,
            'A_mean': float(A_mean),
            'A_std': float(A_std)
        }
        
        return normalized_data, normalization_params
        
    def denormalize_predictions(self, 
                              predictions: torch.Tensor,
                              normalization_params: Dict[str, float]) -> torch.Tensor:
        """
        Denormalize model predictions back to original scale
        
        Args:
            predictions: Normalized predictions
            normalization_params: Normalization parameters
            
        Returns:
            Denormalized predictions
        """
        A_mean = normalization_params['A_mean']
        A_std = normalization_params['A_std']
        
        return predictions * A_std + A_mean
        
    def get_data_statistics(self, absorbance_data: torch.Tensor) -> Dict[str, float]:
        """Get comprehensive data statistics"""
        stats = {
            'shape': tuple(absorbance_data.shape),
            'min_absorbance': float(absorbance_data.min()),
            'max_absorbance': float(absorbance_data.max()),
            'mean_absorbance': float(absorbance_data.mean()),
            'std_absorbance': float(absorbance_data.std()),
            'concentration_values': self.concentration_values.cpu().tolist(),
            'wavelength_range': [float(self.wavelengths.min()), float(self.wavelengths.max())],
            'n_wavelengths': self.n_wavelengths,
            'device': str(self.device)
        }
        
        return stats
        
    def save_data(self, absorbance_data: torch.Tensor, filepath: str):
        """Save spectral data to file"""
        # Convert to numpy for saving
        wavelengths_np = self.wavelengths.cpu().numpy()
        concentrations_np = self.concentration_values.cpu().numpy()
        absorbance_np = absorbance_data.cpu().numpy()
        
        # Create DataFrame
        df = pd.DataFrame(
            absorbance_np.T,
            index=wavelengths_np,
            columns=[f"{c}ppb" for c in concentrations_np]
        )
        df.index.name = 'wavelength_nm'
        
        # Save to CSV
        df.to_csv(filepath)
        print(f"💾 Spectral data saved to {filepath}")
        
    def load_from_existing_format(self, base_data_dir: str) -> torch.Tensor:
        """
        Load data from existing Geodesic NODE format
        
        Args:
            base_data_dir: Path to base Geodesic_NODE directory
            
        Returns:
            Absorbance tensor [6, 601]
        """
        data_path = Path(base_data_dir) / "data" / "0.30MB_AuNP_As.csv"
        
        if data_path.exists():
            return self.load_real_data(str(data_path))
        else:
            print(f"⚠️  Data file not found at {data_path}, using synthetic data")
            return self.generate_synthetic_data()


def create_full_dataset(device: torch.device = torch.device('mps'),
                       use_synthetic: bool = True,
                       data_path: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create complete spectral dataset for training
    
    Args:
        device: Target device
        use_synthetic: Whether to use synthetic data
        data_path: Path to real data file
        
    Returns:
        Tuple of (wavelengths, absorbance_data)
    """
    generator = SpectralDataGenerator(device=device)
    
    if use_synthetic or data_path is None:
        absorbance_data = generator.generate_synthetic_data(noise_level=0.02)
    else:
        absorbance_data = generator.load_real_data(data_path)
        
    return generator.wavelengths, absorbance_data