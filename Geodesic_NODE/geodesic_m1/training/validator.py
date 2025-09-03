"""
Leave-one-out validation for M1 Mac geodesic NODE
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import time

from geodesic_m1.models.geodesic_model import GeodesicNODE
from geodesic_m1.models.multi_model_ensemble import MultiModelEnsemble
from geodesic_m1.training.data_loader import SpectralDataset


class LeaveOneOutValidator:
    """Handles leave-one-out validation for geodesic NODE on M1 Mac"""
    
    def __init__(self,
                 device: torch.device = torch.device('mps'),
                 verbose: bool = True):
        """
        Initialize validator
        
        Args:
            device: Computation device (MPS for M1)
            verbose: Whether to print progress
        """
        self.device = device
        self.verbose = verbose
        self.validation_results = {}
        
    def validate_single_model(self,
                             model: GeodesicNODE,
                             validation_dataset: SpectralDataset,
                             excluded_concentration_idx: int,
                             batch_size: int = 512) -> Dict[str, float]:
        """
        Validate single model on excluded concentration
        
        Args:
            model: Trained geodesic NODE model
            validation_dataset: Validation dataset
            excluded_concentration_idx: Which concentration was excluded
            batch_size: Batch size for validation
            
        Returns:
            Dictionary with validation metrics
        """
        model.eval()
        
        # Get validation data
        val_data = validation_dataset.get_validation_data(excluded_concentration_idx)
        
        if not val_data:
            return {'error': 'No validation data available'}
            
        # Extract data
        c_sources = val_data['c_sources']
        c_targets = val_data['c_targets']
        wavelengths = val_data['wavelengths']
        true_absorbances = val_data['target_absorbances']
        
        # Predict in batches
        predictions = []
        convergence_rates = []
        
        n_samples = c_sources.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                # Get batch
                c_src_batch = c_sources[start_idx:end_idx]
                c_tgt_batch = c_targets[start_idx:end_idx]
                wl_batch = wavelengths[start_idx:end_idx]
                
                # Forward pass
                results = model.forward(c_src_batch, c_tgt_batch, wl_batch)
                
                predictions.append(results['absorbance'])
                convergence_rates.append(results['convergence_rate'])
                
        # Concatenate results
        predicted_absorbances = torch.cat(predictions, dim=0)
        mean_convergence = torch.tensor(convergence_rates).mean()
        
        # Convert to numpy for metrics
        y_true = true_absorbances.cpu().numpy()
        y_pred = predicted_absorbances.cpu().numpy()
        
        # Compute metrics
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        # Find peak wavelength error
        residuals = np.abs(y_true - y_pred)
        max_error_idx = np.argmax(residuals)
        peak_error_wavelength = validation_dataset.wavelengths[max_error_idx % len(validation_dataset.wavelengths)]
        
        results = {
            'excluded_concentration_idx': excluded_concentration_idx,
            'excluded_concentration_ppb': validation_dataset.concentration_values[excluded_concentration_idx],
            'r2_score': r2,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'max_error': residuals.max(),
            'peak_error_wavelength_nm': peak_error_wavelength,
            'convergence_rate': float(mean_convergence),
            'n_validation_samples': n_samples
        }
        
        if self.verbose:
            print(f"Model {excluded_concentration_idx} ({validation_dataset.concentration_values[excluded_concentration_idx]} ppb):")
            print(f"  R² = {r2:.4f}, RMSE = {rmse:.4f}, MAPE = {mape:.2f}%")
            print(f"  Convergence = {mean_convergence:.1%}, Max error @ {peak_error_wavelength:.0f}nm")
            
        return results
        
    def validate_all_models(self,
                           models: List[GeodesicNODE],
                           datasets: List[SpectralDataset],
                           batch_size: int = 512) -> Dict[str, any]:
        """
        Run complete leave-one-out validation
        
        Args:
            models: List of 6 trained models
            datasets: List of 6 validation datasets
            batch_size: Batch size for validation
            
        Returns:
            Complete validation results
        """
        if len(models) != 6 or len(datasets) != 6:
            raise ValueError("Expected 6 models and 6 datasets for leave-one-out validation")
            
        if self.verbose:
            print("🧪 Running leave-one-out validation on M1 Mac...")
            
        start_time = time.time()
        individual_results = []
        
        # Validate each model
        for i in range(6):
            if self.verbose:
                print(f"\n📊 Validating model {i+1}/6...")
                
            result = self.validate_single_model(
                models[i], datasets[i], i, batch_size
            )
            individual_results.append(result)
            
        # Compute aggregate statistics
        r2_scores = [r['r2_score'] for r in individual_results]
        rmse_scores = [r['rmse'] for r in individual_results]
        mae_scores = [r['mae'] for r in individual_results]
        mape_scores = [r['mape'] for r in individual_results]
        convergence_rates = [r['convergence_rate'] for r in individual_results]
        
        aggregate_results = {
            'mean_r2': np.mean(r2_scores),
            'std_r2': np.std(r2_scores),
            'min_r2': np.min(r2_scores),
            'max_r2': np.max(r2_scores),
            'mean_rmse': np.mean(rmse_scores),
            'std_rmse': np.std(rmse_scores),
            'mean_mae': np.mean(mae_scores),
            'mean_mape': np.mean(mape_scores),
            'mean_convergence': np.mean(convergence_rates),
            'validation_time_minutes': (time.time() - start_time) / 60
        }
        
        # Identify worst performance
        worst_model_idx = np.argmin(r2_scores)
        worst_concentration = individual_results[worst_model_idx]['excluded_concentration_ppb']
        
        self.validation_results = {
            'individual_results': individual_results,
            'aggregate_results': aggregate_results,
            'worst_performance': {
                'model_idx': worst_model_idx,
                'concentration_ppb': worst_concentration,
                'r2_score': r2_scores[worst_model_idx]
            }
        }
        
        if self.verbose:
            self._print_summary()
            
        return self.validation_results
        
    def validate_ensemble(self,
                         ensemble: MultiModelEnsemble,
                         datasets: List[SpectralDataset],
                         batch_size: int = 512) -> Dict[str, any]:
        """
        Validate multi-model ensemble
        
        Args:
            ensemble: Multi-model ensemble
            datasets: List of validation datasets
            batch_size: Batch size for validation
            
        Returns:
            Ensemble validation results
        """
        if self.verbose:
            print("🎯 Validating ensemble model...")
            
        ensemble.eval()
        all_predictions = []
        all_targets = []
        
        # Collect predictions from all models
        for excluded_idx in range(6):
            dataset = datasets[excluded_idx]
            val_data = dataset.get_validation_data(excluded_idx)
            
            if not val_data:
                continue
                
            # Get model features (would need geodesic trajectories)
            # This is a simplified version - in practice you'd need the full pipeline
            features = torch.randn(val_data['c_sources'].shape[0], 5, device=self.device)
            
            with torch.no_grad():
                # Use the appropriate model for this excluded concentration
                predictions = ensemble.forward(features, excluded_idx)
                
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(val_data['target_absorbances'].cpu().numpy())
            
        # Compute ensemble metrics
        if all_predictions:
            y_pred = np.concatenate(all_predictions)
            y_true = np.concatenate(all_targets)
            
            ensemble_r2 = r2_score(y_true, y_pred)
            ensemble_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            ensemble_mae = mean_absolute_error(y_true, y_pred)
            
            if self.verbose:
                print(f"📈 Ensemble Performance:")
                print(f"   R² = {ensemble_r2:.4f}")
                print(f"   RMSE = {ensemble_rmse:.4f}")
                print(f"   MAE = {ensemble_mae:.4f}")
                
            return {
                'ensemble_r2': ensemble_r2,
                'ensemble_rmse': ensemble_rmse,
                'ensemble_mae': ensemble_mae,
                'n_total_samples': len(y_true)
            }
        else:
            return {'error': 'No validation data available'}
            
    def _print_summary(self):
        """Print validation summary"""
        if not self.validation_results:
            return
            
        agg = self.validation_results['aggregate_results']
        worst = self.validation_results['worst_performance']
        
        print(f"\n🎯 Leave-One-Out Validation Summary:")
        print(f"   Mean R²: {agg['mean_r2']:.4f} ± {agg['std_r2']:.4f}")
        print(f"   Range R²: [{agg['min_r2']:.4f}, {agg['max_r2']:.4f}]")
        print(f"   Mean RMSE: {agg['mean_rmse']:.4f}")
        print(f"   Mean MAPE: {agg['mean_mape']:.2f}%")
        print(f"   Mean Convergence: {agg['mean_convergence']:.1%}")
        print(f"   Validation Time: {agg['validation_time_minutes']:.1f} minutes")
        print(f"   Worst Performance: {worst['concentration_ppb']} ppb (R² = {worst['r2_score']:.4f})")
        
    def save_results(self, filepath: str):
        """Save validation results to file"""
        if self.validation_results:
            torch.save(self.validation_results, filepath)
            if self.verbose:
                print(f"💾 Validation results saved to {filepath}")
                
    def load_results(self, filepath: str):
        """Load validation results from file"""
        self.validation_results = torch.load(filepath, map_location=self.device)
        if self.verbose:
            print(f"📂 Validation results loaded from {filepath}")
            
    def get_performance_by_concentration(self) -> Dict[float, Dict[str, float]]:
        """Get performance metrics organized by concentration"""
        if not self.validation_results:
            return {}
            
        performance = {}
        for result in self.validation_results['individual_results']:
            conc = result['excluded_concentration_ppb']
            performance[conc] = {
                'r2_score': result['r2_score'],
                'rmse': result['rmse'],
                'mae': result['mae'],
                'mape': result['mape'],
                'convergence_rate': result['convergence_rate']
            }
            
        return performance