#!/usr/bin/env python3
"""
Leave-One-Out Cross-Validation for Geodesic Spectral Model
Computes comprehensive metrics and compares to linear baseline
"""

import torch
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, Tuple, List
import os
from tqdm import tqdm

from data_loader import get_holdout_test_data, SpectralDataset
from geodesic_model import GeodesicSpectralModel
from train import Trainer


class Validator:
    """
    Performs Leave-One-Out Cross-Validation and computes metrics
    """
    
    def __init__(self, model: GeodesicSpectralModel = None):
        """
        Initialize validator
        
        Args:
            model: Pre-trained model (if None, will train new models)
        """
        self.model = model
        self.concentrations = [0, 10, 20, 30, 40, 60]  # ppb
        self.metrics_names = [
            'R2', 'RMSE', 'MAE', 'MAPE', 'Max_Error',
            'Peak_Lambda_True', 'Peak_Lambda_Pred', 'Peak_Lambda_Error',
            'Peak_Value_True', 'Peak_Value_Pred', 'Peak_Value_Error',
            'Min_Lambda_True', 'Min_Lambda_Pred', 'Min_Lambda_Error',
            'Min_Value_True', 'Min_Value_Pred', 'Min_Value_Error',
            'Mean_True', 'Mean_Pred', 'Mean_Error',
            'Std_True', 'Std_Pred', 'Std_Error'
        ]
    
    def compute_metrics(self, 
                       y_true: np.ndarray, 
                       y_pred: np.ndarray,
                       wavelengths: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive validation metrics
        
        Args:
            y_true: True absorbance values
            y_pred: Predicted absorbance values
            wavelengths: Wavelength values
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic regression metrics
        metrics['R2'] = r2_score(y_true, y_pred)
        metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['MAE'] = mean_absolute_error(y_true, y_pred)
        
        # MAPE (avoiding division by zero)
        mask = np.abs(y_true) > 1e-10
        if mask.sum() > 0:
            metrics['MAPE'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            metrics['MAPE'] = 0.0
        
        metrics['Max_Error'] = np.max(np.abs(y_true - y_pred))
        
        # Peak analysis
        peak_idx_true = np.argmax(y_true)
        peak_idx_pred = np.argmax(y_pred)
        
        metrics['Peak_Lambda_True'] = wavelengths[peak_idx_true]
        metrics['Peak_Lambda_Pred'] = wavelengths[peak_idx_pred]
        metrics['Peak_Lambda_Error'] = np.abs(metrics['Peak_Lambda_True'] - metrics['Peak_Lambda_Pred'])
        
        metrics['Peak_Value_True'] = y_true[peak_idx_true]
        metrics['Peak_Value_Pred'] = y_pred[peak_idx_true]
        metrics['Peak_Value_Error'] = np.abs(metrics['Peak_Value_True'] - metrics['Peak_Value_Pred'])
        
        # Min analysis
        min_idx_true = np.argmin(y_true)
        min_idx_pred = np.argmin(y_pred)
        
        metrics['Min_Lambda_True'] = wavelengths[min_idx_true]
        metrics['Min_Lambda_Pred'] = wavelengths[min_idx_pred]
        metrics['Min_Lambda_Error'] = np.abs(metrics['Min_Lambda_True'] - metrics['Min_Lambda_Pred'])
        
        metrics['Min_Value_True'] = y_true[min_idx_true]
        metrics['Min_Value_Pred'] = y_pred[min_idx_true]
        metrics['Min_Value_Error'] = np.abs(metrics['Min_Value_True'] - metrics['Min_Value_Pred'])
        
        # Statistical properties
        metrics['Mean_True'] = np.mean(y_true)
        metrics['Mean_Pred'] = np.mean(y_pred)
        metrics['Mean_Error'] = np.abs(metrics['Mean_True'] - metrics['Mean_Pred'])
        
        metrics['Std_True'] = np.std(y_true)
        metrics['Std_Pred'] = np.std(y_pred)
        metrics['Std_Error'] = np.abs(metrics['Std_True'] - metrics['Std_Pred'])
        
        return metrics
    
    def linear_interpolation_baseline(self, 
                                     holdout_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute linear interpolation baseline for comparison
        
        Args:
            holdout_idx: Index of concentration to hold out
        
        Returns:
            wavelengths, true_absorbance, predicted_absorbance
        """
        # Load data
        df = pd.read_csv("0.30MB_AuNP_As.csv")
        wavelengths = df['Wavelength'].values
        concentrations = np.array([float(col) for col in df.columns[1:]])
        absorbance_matrix = df.iloc[:, 1:].values
        
        # Get training concentrations (exclude holdout)
        train_mask = np.ones(len(concentrations), dtype=bool)
        train_mask[holdout_idx] = False
        train_concs = concentrations[train_mask]
        train_abs = absorbance_matrix[:, train_mask]
        
        # True values
        true_abs = absorbance_matrix[:, holdout_idx]
        holdout_conc = concentrations[holdout_idx]
        
        # Linear interpolation for each wavelength
        pred_abs = np.zeros(len(wavelengths))
        for i in range(len(wavelengths)):
            # Use cubic if enough points, otherwise linear
            kind = 'cubic' if len(train_concs) > 3 else 'linear'
            interp_func = interp1d(train_concs, train_abs[i, :], 
                                  kind=kind, fill_value='extrapolate')
            pred_abs[i] = interp_func(holdout_conc)
        
        return wavelengths, true_abs, pred_abs
    
    def validate_holdout(self, holdout_idx: int) -> Dict[str, Dict[str, float]]:
        """
        Validate model on a single held-out concentration
        
        Args:
            holdout_idx: Index of concentration to hold out
        
        Returns:
            Dictionary with 'geodesic' and 'linear' metrics
        """
        print(f"\nValidating holdout concentration: {self.concentrations[holdout_idx]} ppb")
        
        # Get test data
        wavelengths, true_abs, test_conc = get_holdout_test_data(holdout_idx)
        
        # Train new model if not provided
        if self.model is None:
            print("  Training new model (excluding holdout)...")
            model = GeodesicSpectralModel()
            trainer = Trainer(
                model=model,
                n_epochs=100,  # Reduced for validation
                batch_size=32
            )
            trainer.train(exclude_concentration_idx=holdout_idx, save_every=50)
            self.model = model
        
        # Prepare model for evaluation
        self.model.eval()
        dataset = SpectralDataset(exclude_concentration_idx=holdout_idx)
        
        # Get predictions from geodesic model
        print("  Computing geodesic predictions...")
        pred_abs_geodesic = []
        
        # For each wavelength, interpolate using trained model
        with torch.no_grad():
            for wl_idx, wl in enumerate(tqdm(wavelengths, desc="    Wavelengths", leave=False)):
                # Use neighboring concentrations for interpolation
                # Find closest training concentrations
                train_concs = [c for i, c in enumerate(self.concentrations) if i != holdout_idx]
                
                # For simplicity, use all pairs and average
                predictions = []
                for c_source in train_concs:
                    # Normalize inputs
                    c_source_norm = dataset.normalize_concentration(c_source)
                    c_target_norm = dataset.normalize_concentration(test_conc)
                    wl_norm = dataset.normalize_wavelength(wl)
                    
                    # Get prediction
                    c_source_tensor = torch.tensor([c_source_norm], dtype=torch.float32)
                    c_target_tensor = torch.tensor([c_target_norm], dtype=torch.float32)
                    wl_tensor = torch.tensor([wl_norm], dtype=torch.float32)
                    
                    output = self.model(c_source_tensor, c_target_tensor, wl_tensor)
                    abs_norm = output['absorbance'].item()
                    abs_raw = dataset.denormalize_absorbance(abs_norm)
                    predictions.append(abs_raw)
                
                # Average predictions
                pred_abs_geodesic.append(np.mean(predictions))
        
        pred_abs_geodesic = np.array(pred_abs_geodesic)
        
        # Get linear baseline
        print("  Computing linear baseline...")
        _, _, pred_abs_linear = self.linear_interpolation_baseline(holdout_idx)
        
        # Compute metrics
        metrics_geodesic = self.compute_metrics(true_abs, pred_abs_geodesic, wavelengths)
        metrics_linear = self.compute_metrics(true_abs, pred_abs_linear, wavelengths)
        
        print(f"  Geodesic R²: {metrics_geodesic['R2']:.4f}")
        print(f"  Linear R²: {metrics_linear['R2']:.4f}")
        print(f"  Improvement: {metrics_geodesic['R2'] - metrics_linear['R2']:.4f}")
        
        return {
            'geodesic': metrics_geodesic,
            'linear': metrics_linear
        }
    
    def leave_one_out_validation(self) -> pd.DataFrame:
        """
        Perform full leave-one-out cross-validation
        
        Returns:
            DataFrame with all metrics for all concentrations
        """
        print("Starting Leave-One-Out Cross-Validation...")
        print(f"Concentrations: {self.concentrations} ppb")
        
        results = []
        
        for holdout_idx in range(len(self.concentrations)):
            metrics = self.validate_holdout(holdout_idx)
            
            # Create row for results
            row = {
                'Concentration_ppb': self.concentrations[holdout_idx],
                'Holdout_Index': holdout_idx
            }
            
            # Add geodesic metrics
            for metric_name, value in metrics['geodesic'].items():
                row[f'Geodesic_{metric_name}'] = value
            
            # Add linear metrics
            for metric_name, value in metrics['linear'].items():
                row[f'Linear_{metric_name}'] = value
            
            # Add improvements
            for metric_name in metrics['geodesic'].keys():
                if metric_name in ['R2', 'Peak_Lambda_Error', 'Peak_Value_Error']:
                    improvement = metrics['geodesic'][metric_name] - metrics['linear'][metric_name]
                    row[f'Improvement_{metric_name}'] = improvement
            
            results.append(row)
            
            # Reset model for next holdout
            self.model = None
        
        # Create DataFrame
        df_results = pd.DataFrame(results)
        
        # Print summary
        self.print_summary(df_results)
        
        return df_results
    
    def print_summary(self, df: pd.DataFrame):
        """Print validation summary"""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        # Key metrics comparison
        print("\nKey Metrics (Average across all holdouts):")
        print("-"*40)
        
        metrics_to_show = ['R2', 'RMSE', 'MAPE', 'Peak_Lambda_Error']
        
        for metric in metrics_to_show:
            geo_col = f'Geodesic_{metric}'
            lin_col = f'Linear_{metric}'
            
            if geo_col in df.columns and lin_col in df.columns:
                geo_mean = df[geo_col].mean()
                lin_mean = df[lin_col].mean()
                improvement = geo_mean - lin_mean
                
                print(f"\n{metric}:")
                print(f"  Geodesic: {geo_mean:.4f}")
                print(f"  Linear:   {lin_mean:.4f}")
                print(f"  Improvement: {improvement:+.4f}")
        
        # Worst case (60 ppb)
        print("\n" + "-"*40)
        print("60 ppb (Worst Case):")
        worst_case = df[df['Concentration_ppb'] == 60].iloc[0]
        print(f"  Geodesic R²: {worst_case['Geodesic_R2']:.4f}")
        print(f"  Linear R²:   {worst_case['Linear_R2']:.4f}")
        print(f"  Improvement: {worst_case['Geodesic_R2'] - worst_case['Linear_R2']:+.4f}")
        
        # Save results
        df.to_csv('validation_results.csv', index=False)
        print("\nResults saved to validation_results.csv")


def main():
    """Main validation script"""
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create validator
    validator = Validator()
    
    # Run leave-one-out validation
    results = validator.leave_one_out_validation()
    
    print("\nValidation completed!")


if __name__ == "__main__":
    main()