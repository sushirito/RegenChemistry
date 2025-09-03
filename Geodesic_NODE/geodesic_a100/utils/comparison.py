"""
Basic interpolation baseline for comparison with geodesic model
"""

import numpy as np
from scipy.interpolate import interp1d
from typing import Tuple, Dict, List
import pandas as pd


class BasicInterpolation:
    """Perform basic cubic/linear interpolation for spectral data"""
    
    def __init__(self, csv_path: str = 'data/0.30MB_AuNP_As.csv'):
        """
        Initialize with spectral data
        
        Args:
            csv_path: Path to CSV file with spectral data
        """
        # Load data
        df = pd.read_csv(csv_path)
        self.wavelengths = df['Wavelength'].values  # 601 points
        self.concentrations = [float(col) for col in df.columns[1:]]  # [0, 10, 20, 30, 40, 60]
        self.absorbance_matrix = df.iloc[:, 1:].values  # [601, 6]
        
        print(f"Loaded data: {len(self.wavelengths)} wavelengths, {len(self.concentrations)} concentrations")
        
    def interpolate_holdout(self, holdout_idx: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Perform leave-one-out interpolation
        
        Args:
            holdout_idx: Index of concentration to hold out (0-5)
            
        Returns:
            predictions: Interpolated absorbance values [n_wavelengths]
            actual: Actual absorbance values [n_wavelengths]
            metrics: Dictionary with performance metrics
        """
        # Get training concentrations (exclude holdout)
        train_indices = [i for i in range(len(self.concentrations)) if i != holdout_idx]
        train_concs = [self.concentrations[i] for i in train_indices]
        train_abs = self.absorbance_matrix[:, train_indices]  # [601, 5]
        
        # Get holdout data
        holdout_conc = self.concentrations[holdout_idx]
        actual_abs = self.absorbance_matrix[:, holdout_idx]  # [601]
        
        # Interpolate for each wavelength
        interpolated_abs = np.zeros(len(self.wavelengths))
        
        for i in range(len(self.wavelengths)):
            # Use cubic if we have enough points, otherwise linear
            if len(train_concs) >= 4:
                interp_func = interp1d(train_concs, train_abs[i, :], 
                                      kind='cubic', fill_value='extrapolate')
            else:
                interp_func = interp1d(train_concs, train_abs[i, :], 
                                      kind='linear', fill_value='extrapolate')
            
            interpolated_abs[i] = interp_func(holdout_conc)
        
        # Calculate metrics
        metrics = self.calculate_metrics(interpolated_abs, actual_abs)
        
        return interpolated_abs, actual_abs, metrics
    
    def calculate_metrics(self, predictions: np.ndarray, actual: np.ndarray) -> Dict:
        """
        Calculate performance metrics
        
        Args:
            predictions: Predicted values
            actual: Actual values
            
        Returns:
            Dictionary with MSE, MAE, R² scores
        """
        # MSE
        mse = np.mean((predictions - actual) ** 2)
        
        # MAE
        mae = np.mean(np.abs(predictions - actual))
        
        # R² score
        ss_res = np.sum((actual - predictions) ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else float('-inf')
        
        # Peak wavelength error (wavelength with max absorbance)
        peak_idx_actual = np.argmax(actual)
        peak_idx_pred = np.argmax(predictions)
        peak_error = abs(self.wavelengths[peak_idx_actual] - self.wavelengths[peak_idx_pred])
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'peak_error_nm': peak_error,
            'rmse': np.sqrt(mse)
        }
    
    def get_worst_case_performance(self) -> Tuple[int, Dict]:
        """
        Find the worst performing concentration for interpolation
        Typically this is 60 ppb (index 5) for this dataset
        
        Returns:
            worst_idx: Index of worst performing concentration
            metrics: Performance metrics for worst case
        """
        worst_r2 = float('inf')
        worst_idx = -1
        worst_metrics = None
        
        for idx in range(len(self.concentrations)):
            _, _, metrics = self.interpolate_holdout(idx)
            print(f"  {self.concentrations[idx]} ppb: R²={metrics['r2']:.3f}, MSE={metrics['mse']:.4f}")
            
            if metrics['r2'] < worst_r2:
                worst_r2 = metrics['r2']
                worst_idx = idx
                worst_metrics = metrics
        
        print(f"\nWorst case: {self.concentrations[worst_idx]} ppb with R²={worst_r2:.3f}")
        return worst_idx, worst_metrics