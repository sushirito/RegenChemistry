"""
Comprehensive validation and evaluation pipeline
Compares geodesic model with basic interpolation
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import os
import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from geodesic_m1.utils.metrics import (
    compute_all_metrics,
    interpolate_holdout,
    compute_improvement
)
from geodesic_m1.models.geodesic_model import GeodesicNODE
from geodesic_m1.training.data_loader import SpectralDataset


def validate_single_holdout(
    model: GeodesicNODE,
    dataset: SpectralDataset,
    holdout_idx: int,
    holdout_conc: float,
    wavelengths: np.ndarray,
    absorbance_matrix: np.ndarray,
    concentrations: list,
    device: torch.device,
    batch_size: int = 128
) -> Dict:
    """
    Validate a single model on its holdout concentration
    
    Returns:
        Dictionary with predictions and actual values
    """
    # Get actual spectrum for holdout
    actual = absorbance_matrix[:, holdout_idx]
    
    # Find nearest training concentration as source
    train_concs = [concentrations[i] for i in range(len(concentrations)) if i != holdout_idx]
    nearest_idx = np.argmin([abs(tc - holdout_conc) for tc in train_concs])
    source_c = train_concs[nearest_idx]
    
    # Normalize concentrations
    c_source_norm = (source_c - dataset.c_mean) / dataset.c_std
    c_target_norm = (holdout_conc - dataset.c_mean) / dataset.c_std
    
    # Predict spectrum using geodesic model
    predictions = []
    model.eval()
    
    with torch.no_grad():
        for i in range(0, len(wavelengths), batch_size):
            end = min(i + batch_size, len(wavelengths))
            batch_wl = wavelengths[i:end]
            
            # Normalize wavelengths
            wl_norm = (batch_wl - dataset.lambda_mean) / dataset.lambda_std
            
            # Create batch
            c_sources = torch.full((end - i,), c_source_norm, device=device, dtype=torch.float32)
            c_targets = torch.full((end - i,), c_target_norm, device=device, dtype=torch.float32)
            wl_tensor = torch.tensor(wl_norm, dtype=torch.float32, device=device)
            
            # Forward pass
            output = model(c_sources, c_targets, wl_tensor)
            
            # Denormalize predictions
            pred = output['absorbance'].cpu().numpy()
            pred = pred * dataset.A_std + dataset.A_mean
            predictions.extend(pred)
    
    predictions = np.array(predictions)
    
    return {
        'concentration': holdout_conc,
        'predictions': predictions,
        'actual': actual,
        'wavelengths': wavelengths
    }


def run_complete_validation(
    model_paths: Dict[int, str],
    data_path: str,
    device: torch.device,
    save_metrics: bool = True,
    save_predictions: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run complete validation with all 6 models and compute 20+ metrics
    
    Args:
        model_paths: Dictionary mapping concentration index to model path
        data_path: Path to spectral data CSV
        device: Computation device
        save_metrics: Whether to save metrics to CSV
        save_predictions: Whether to save predictions to CSV
        
    Returns:
        Tuple of (metrics_df, predictions_df)
    """
    # Load data
    df = pd.read_csv(data_path)
    wavelengths = df['Wavelength'].values
    concentrations = [float(col) for col in df.columns[1:]]
    absorbance_matrix = df.iloc[:, 1:].values
    
    # Storage for results
    metrics_rows = []
    predictions_rows = []
    
    print("\n🔬 Running complete validation...")
    
    for holdout_idx, holdout_conc in enumerate(concentrations):
        print(f"\n📍 Validating holdout {holdout_conc} ppb...")
        
        # Load model for this holdout
        if holdout_idx not in model_paths:
            print(f"   ⚠️  No model found for holdout {holdout_idx}, skipping...")
            continue
            
        model_path = model_paths[holdout_idx]
        
        # Create dataset for normalization
        dataset = SpectralDataset(
            concentration_values=concentrations,
            wavelengths=wavelengths,
            absorbance_data=absorbance_matrix,
            excluded_concentration_idx=holdout_idx,
            normalize=True,
            device=device
        )
        
        # Load model with data for absorbance lookup
        model = GeodesicNODE(
            metric_hidden_dims=[128, 256],
            flow_hidden_dims=[64, 128],
            n_trajectory_points=50,
            shooting_max_iter=10,
            shooting_tolerance=1e-4,
            shooting_learning_rate=0.5,
            christoffel_grid_size=(500, 601),
            device=device,
            use_adjoint=False,
            concentrations=np.array(concentrations),
            wavelengths=wavelengths,
            absorbance_matrix=absorbance_matrix
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Get geodesic predictions
        geo_results = validate_single_holdout(
            model, dataset, holdout_idx, holdout_conc,
            wavelengths, absorbance_matrix, concentrations,
            device
        )
        
        # Get basic interpolation predictions
        basic_pred = interpolate_holdout(
            wavelengths, concentrations, absorbance_matrix,
            holdout_idx, holdout_conc
        )
        
        actual = geo_results['actual']
        geodesic_pred = geo_results['predictions']
        
        # Compute metrics for both methods
        geo_metrics = compute_all_metrics(actual, geodesic_pred, wavelengths)
        geo_metrics['Concentration_ppb'] = holdout_conc
        geo_metrics['Method'] = 'Geodesic'
        
        basic_metrics = compute_all_metrics(actual, basic_pred, wavelengths)
        basic_metrics['Concentration_ppb'] = holdout_conc
        basic_metrics['Method'] = 'Basic'
        
        metrics_rows.append(geo_metrics)
        metrics_rows.append(basic_metrics)
        
        # Store predictions
        for i, wl in enumerate(wavelengths):
            predictions_rows.append({
                'Concentration_ppb': holdout_conc,
                'Wavelength_nm': wl,
                'Actual': actual[i],
                'BasicInterp': basic_pred[i],
                'Geodesic': geodesic_pred[i]
            })
        
        # Print summary
        print(f"   🎯 Geodesic R²: {geo_metrics['R2_Score']:.4f}")
        print(f"   📏 Basic R²: {basic_metrics['R2_Score']:.4f}")
        print(f"   📈 Improvement: {geo_metrics['R2_Score'] - basic_metrics['R2_Score']:+.4f}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    # Create DataFrames
    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.DataFrame(predictions_rows)
    
    # Reorder columns for metrics
    metric_cols = ['Concentration_ppb', 'Method'] + \
                  [col for col in metrics_df.columns if col not in ['Concentration_ppb', 'Method']]
    metrics_df = metrics_df[metric_cols]
    
    # Save to CSV if requested
    if save_metrics:
        metrics_path = 'outputs/metrics/validation_metrics_20plus.csv'
        metrics_df.to_csv(metrics_path, index=False, float_format='%.6f')
        print(f"\n💾 Metrics saved to {metrics_path}")
    
    if save_predictions:
        pred_path = 'outputs/predictions/validation_predictions.csv'
        predictions_df.to_csv(pred_path, index=False, float_format='%.6f')
        print(f"💾 Predictions saved to {pred_path}")
    
    # Print summary statistics
    print("\n📊 VALIDATION SUMMARY:")
    
    geo_df = metrics_df[metrics_df['Method'] == 'Geodesic']
    basic_df = metrics_df[metrics_df['Method'] == 'Basic']
    
    print(f"   Geodesic Mean R²: {geo_df['R2_Score'].mean():.4f} ± {geo_df['R2_Score'].std():.4f}")
    print(f"   Basic Mean R²: {basic_df['R2_Score'].mean():.4f} ± {basic_df['R2_Score'].std():.4f}")
    print(f"   Mean Improvement: {(geo_df['R2_Score'].mean() - basic_df['R2_Score'].mean()):.4f}")
    
    print(f"\n   Geodesic Mean RMSE: {geo_df['RMSE'].mean():.4f}")
    print(f"   Basic Mean RMSE: {basic_df['RMSE'].mean():.4f}")
    
    return metrics_df, predictions_df


def compute_improvement_table(metrics_df: pd.DataFrame, save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Compute improvement percentages for all metrics
    
    Args:
        metrics_df: DataFrame with metrics for both methods
        save_path: Optional path to save improvement table
        
    Returns:
        DataFrame with improvements
    """
    improvements = []
    
    concentrations = metrics_df['Concentration_ppb'].unique()
    
    for conc in concentrations:
        geo_row = metrics_df[(metrics_df['Concentration_ppb'] == conc) & 
                            (metrics_df['Method'] == 'Geodesic')].iloc[0]
        basic_row = metrics_df[(metrics_df['Concentration_ppb'] == conc) & 
                              (metrics_df['Method'] == 'Basic')].iloc[0]
        
        for metric in metrics_df.columns:
            if metric in ['Concentration_ppb', 'Method']:
                continue
                
            geo_val = geo_row[metric]
            basic_val = basic_row[metric]
            
            improvement = compute_improvement(basic_val, geo_val, metric)
            
            improvements.append({
                'Concentration_ppb': conc,
                'Metric': metric,
                'Basic': basic_val,
                'Geodesic': geo_val,
                'Improvement_%': improvement
            })
    
    improvement_df = pd.DataFrame(improvements)
    
    if save_path:
        improvement_df.to_csv(save_path, index=False, float_format='%.2f')
        print(f"💾 Improvement table saved to {save_path}")
    
    return improvement_df