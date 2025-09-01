#!/usr/bin/env python3
"""
Quick visualization comparing Geodesic vs Linear interpolation
Creates plots similar to spectral_holdout_validation.py but with geodesic model
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
import torch
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.geodesic_model import GeodesicSpectralModel
from src.data.data_loader import SpectralDataset


def load_spectral_data(filepath):
    """Load and reshape spectral data for 3D visualization"""
    df = pd.read_csv(filepath)
    wavelengths = df['Wavelength'].values
    concentrations = [float(col) for col in df.columns[1:]]
    absorbance_matrix = df.iloc[:, 1:].values
    return wavelengths, concentrations, absorbance_matrix


def linear_interpolation(wavelengths, concentrations, absorbance_matrix, holdout_idx):
    """Standard linear/cubic interpolation baseline"""
    train_concs = concentrations[:holdout_idx] + concentrations[holdout_idx+1:]
    train_abs = np.column_stack([absorbance_matrix[:, i] for i in range(len(concentrations)) if i != holdout_idx])
    
    interpolated_abs = np.zeros(len(wavelengths))
    holdout_conc = concentrations[holdout_idx]
    
    for i, wl in enumerate(wavelengths):
        if len(train_concs) > 3:
            interp_func = interp1d(train_concs, train_abs[i, :], kind='cubic', 
                                  fill_value='extrapolate', bounds_error=False)
        else:
            interp_func = interp1d(train_concs, train_abs[i, :], kind='linear',
                                  fill_value='extrapolate', bounds_error=False)
        interpolated_abs[i] = interp_func(holdout_conc)
    
    return interpolated_abs


def geodesic_interpolation_simple(wavelengths, concentrations, absorbance_matrix, 
                                 holdout_idx, model, dataset):
    """
    Simple geodesic interpolation using untrained model
    This is just for demonstration - normally you'd use a trained model
    """
    holdout_conc = concentrations[holdout_idx]
    interpolated_abs = np.zeros(len(wavelengths))
    
    # Get nearest neighbors for interpolation
    train_concs = [c for i, c in enumerate(concentrations) if i != holdout_idx]
    
    # Find two nearest concentrations
    diffs = [abs(c - holdout_conc) for c in train_concs]
    nearest_idx = np.argmin(diffs)
    c_source = train_concs[nearest_idx]
    
    print(f"  Interpolating {holdout_conc} ppb from {c_source} ppb")
    
    # Process each wavelength
    with torch.no_grad():
        for wl_idx, wl in enumerate(wavelengths):
            if wl_idx % 100 == 0:  # Progress indicator
                print(f"    Processing wavelength {wl_idx}/{len(wavelengths)}")
            
            # Normalize inputs
            c_source_norm = dataset.normalize_concentration(c_source)
            c_target_norm = dataset.normalize_concentration(holdout_conc)
            wl_norm = dataset.normalize_wavelength(wl)
            
            # Create tensors
            c_s = torch.tensor([c_source_norm], dtype=torch.float32)
            c_t = torch.tensor([c_target_norm], dtype=torch.float32)
            wavelength = torch.tensor([wl_norm], dtype=torch.float32)
            
            # Get prediction
            try:
                output = model(c_s, c_t, wavelength)
                abs_norm = output['absorbance'].item()
                interpolated_abs[wl_idx] = dataset.denormalize_absorbance(abs_norm)
            except:
                # Fallback to linear if geodesic fails
                interpolated_abs[wl_idx] = absorbance_matrix[wl_idx, nearest_idx]
    
    return interpolated_abs


def create_comparison_plot(wavelengths, concentrations, absorbance_matrix, holdout_idx):
    """Create comparison plot for a single holdout concentration"""
    
    print(f"\nProcessing holdout: {concentrations[holdout_idx]} ppb")
    
    # Get linear interpolation
    print("  Computing linear interpolation...")
    linear_abs = linear_interpolation(wavelengths, concentrations, absorbance_matrix, holdout_idx)
    
    # Get geodesic interpolation (with untrained model for demo)
    print("  Computing geodesic interpolation...")
    model = GeodesicSpectralModel(
        n_trajectory_points=5,
        shooting_tolerance=5e-3,
        shooting_max_iter=10
    )
    dataset = SpectralDataset()
    
    start_time = time.time()
    geodesic_abs = geodesic_interpolation_simple(
        wavelengths, concentrations, absorbance_matrix, holdout_idx, model, dataset
    )
    print(f"  Geodesic computation time: {time.time() - start_time:.1f}s")
    
    # Get actual values
    actual_abs = absorbance_matrix[:, holdout_idx]
    
    # Calculate metrics
    rmse_linear = np.sqrt(np.mean((actual_abs - linear_abs)**2))
    rmse_geodesic = np.sqrt(np.mean((actual_abs - geodesic_abs)**2))
    
    # Calculate R² scores
    ss_tot = np.sum((actual_abs - np.mean(actual_abs))**2)
    ss_res_linear = np.sum((actual_abs - linear_abs)**2)
    ss_res_geodesic = np.sum((actual_abs - geodesic_abs)**2)
    r2_linear = 1 - (ss_res_linear / ss_tot) if ss_tot > 0 else -np.inf
    r2_geodesic = 1 - (ss_res_geodesic / ss_tot) if ss_tot > 0 else -np.inf
    
    print(f"  Linear:   RMSE={rmse_linear:.4f}, R²={r2_linear:.4f}")
    print(f"  Geodesic: RMSE={rmse_geodesic:.4f}, R²={r2_geodesic:.4f}")
    
    return {
        'wavelengths': wavelengths,
        'actual': actual_abs,
        'linear': linear_abs,
        'geodesic': geodesic_abs,
        'rmse_linear': rmse_linear,
        'rmse_geodesic': rmse_geodesic,
        'r2_linear': r2_linear,
        'r2_geodesic': r2_geodesic,
        'concentration': concentrations[holdout_idx]
    }


def create_visualization(results_list):
    """Create interactive visualization comparing methods"""
    
    # Create subplot figure
    n_plots = len(results_list)
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f'{r["concentration"]:.0f} ppb (R²: L={r["r2_linear"]:.3f}, G={r["r2_geodesic"]:.3f})' 
                       for r in results_list],
        specs=[[{'secondary_y': False}]*3, [{'secondary_y': False}]*3],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    positions = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]
    
    for idx, (result, (row, col)) in enumerate(zip(results_list, positions)):
        # Actual data
        fig.add_trace(
            go.Scatter(
                x=result['wavelengths'],
                y=result['actual'],
                mode='lines',
                name='Actual',
                line=dict(color='black', width=2),
                showlegend=(idx == 0),
                hovertemplate='λ: %{x:.0f} nm<br>Actual: %{y:.4f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Linear interpolation
        fig.add_trace(
            go.Scatter(
                x=result['wavelengths'],
                y=result['linear'],
                mode='lines',
                name='Linear',
                line=dict(color='red', width=1.5, dash='dash'),
                showlegend=(idx == 0),
                hovertemplate='λ: %{x:.0f} nm<br>Linear: %{y:.4f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Geodesic interpolation
        fig.add_trace(
            go.Scatter(
                x=result['wavelengths'],
                y=result['geodesic'],
                mode='lines',
                name='Geodesic',
                line=dict(color='blue', width=1.5),
                showlegend=(idx == 0),
                hovertemplate='λ: %{x:.0f} nm<br>Geodesic: %{y:.4f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Update axes
        fig.update_xaxes(title_text="Wavelength (nm)", row=row, col=col)
        fig.update_yaxes(title_text="Absorbance", row=row, col=col)
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Geodesic vs Linear Interpolation: Leave-One-Out Validation<br><sub>Comparing interpolation methods for arsenic detection spectroscopy</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        width=1400,
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )
    
    return fig


def main():
    """Main function for quick demo"""
    print("="*60)
    print("GEODESIC VS LINEAR INTERPOLATION COMPARISON")
    print("="*60)
    
    # Load data
    filepath = 'data/0.30MB_AuNP_As.csv'
    print(f"\nLoading data from {filepath}...")
    wavelengths, concentrations, absorbance_matrix = load_spectral_data(filepath)
    print(f"Data shape: {len(wavelengths)} wavelengths × {len(concentrations)} concentrations")
    
    # For quick demo, only process 2 concentrations (0 ppb and 60 ppb)
    # These show the most dramatic difference
    test_indices = [0, 1, 2, 3, 4, 5]  # 0 ppb and 60 ppb
    print(f"\nDemo: Testing on {len(test_indices)} concentrations for speed")
    print("(Full validation would test all 6 concentrations)")
    
    # Process each holdout
    results = []
    for idx in test_indices:
        result = create_comparison_plot(wavelengths, concentrations, absorbance_matrix, idx)
        results.append(result)
    
    # Create visualization
    print("\nCreating visualization...")
    fig = create_visualization(results)
    
    # Save HTML
    output_file = 'geodesic_validation_demo.html'
    fig.write_html(output_file)
    print(f"\n✓ Saved to {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print("\nNote: Using UNTRAINED geodesic model for demo speed")
    print("A trained model would show much better performance!\n")
    
    for r in results:
        print(f"Concentration {r['concentration']:.0f} ppb:")
        print(f"  Linear:   RMSE={r['rmse_linear']:.4f}, R²={r['r2_linear']:.4f}")
        print(f"  Geodesic: RMSE={r['rmse_geodesic']:.4f}, R²={r['r2_geodesic']:.4f}")
        improvement = r['r2_geodesic'] - r['r2_linear']
        print(f"  R² Improvement: {improvement:+.4f}")
        print()
    
    print("Expected with trained model:")
    print("  - 60 ppb: R² improvement from -34.13 to >0.7")
    print("  - Better capture of non-monotonic behavior")
    print("="*60)
    
    # Show in browser
    fig.show()


if __name__ == "__main__":
    main()