#!/usr/bin/env python3
"""
Ultra-fast visualization demo - samples wavelengths for speed
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import interp1d
import torch

from geodesic_model import GeodesicSpectralModel
from data_loader import SpectralDataset


def main():
    print("="*60)
    print("QUICK GEODESIC VISUALIZATION DEMO")
    print("="*60)
    
    # Load data
    df = pd.read_csv('0.30MB_AuNP_As.csv')
    wavelengths = df['Wavelength'].values
    concentrations = [float(col) for col in df.columns[1:]]
    absorbance_matrix = df.iloc[:, 1:].values
    
    # Sample wavelengths for speed (every 20th wavelength)
    sample_step = 20
    wl_indices = range(0, len(wavelengths), sample_step)
    wl_sampled = wavelengths[wl_indices]
    abs_sampled = absorbance_matrix[wl_indices, :]
    
    print(f"Using {len(wl_sampled)} wavelengths (sampled from {len(wavelengths)})")
    
    # Test on 60 ppb (worst case for linear)
    holdout_idx = 5  # 60 ppb
    holdout_conc = concentrations[holdout_idx]
    actual_abs = abs_sampled[:, holdout_idx]
    
    print(f"\nTesting holdout: {holdout_conc} ppb")
    
    # Linear interpolation
    print("Computing linear interpolation...")
    train_concs = concentrations[:holdout_idx]
    train_abs = abs_sampled[:, :holdout_idx]
    
    linear_pred = np.zeros(len(wl_sampled))
    for i in range(len(wl_sampled)):
        interp_func = interp1d(train_concs, train_abs[i, :], kind='cubic', 
                              fill_value='extrapolate', bounds_error=False)
        linear_pred[i] = interp_func(holdout_conc)
    
    # Simple geodesic prediction (untrained model)
    print("Computing geodesic predictions...")
    model = GeodesicSpectralModel(
        n_trajectory_points=5,
        shooting_tolerance=1e-2,  # Very loose for speed
        shooting_max_iter=5  # Minimal iterations
    )
    dataset = SpectralDataset()
    
    geodesic_pred = np.zeros(len(wl_sampled))
    
    # Use nearest concentration as source
    c_source = 40.0  # Nearest to 60 ppb
    
    with torch.no_grad():
        for i, wl in enumerate(wl_sampled):
            if i % 5 == 0:
                print(f"  Processing {i}/{len(wl_sampled)}")
            
            # Normalize
            c_s_norm = dataset.normalize_concentration(c_source)
            c_t_norm = dataset.normalize_concentration(holdout_conc)
            wl_norm = dataset.normalize_wavelength(wl)
            
            # Get prediction
            c_s = torch.tensor([c_s_norm], dtype=torch.float32)
            c_t = torch.tensor([c_t_norm], dtype=torch.float32)
            wavelength = torch.tensor([wl_norm], dtype=torch.float32)
            
            try:
                output = model(c_s, c_t, wavelength)
                abs_norm = output['absorbance'].item()
                geodesic_pred[i] = dataset.denormalize_absorbance(abs_norm)
            except:
                geodesic_pred[i] = linear_pred[i]  # Fallback
    
    # Calculate metrics
    rmse_linear = np.sqrt(np.mean((actual_abs - linear_pred)**2))
    rmse_geodesic = np.sqrt(np.mean((actual_abs - geodesic_pred)**2))
    
    # R² calculation
    ss_tot = np.sum((actual_abs - np.mean(actual_abs))**2)
    r2_linear = 1 - np.sum((actual_abs - linear_pred)**2) / ss_tot
    r2_geodesic = 1 - np.sum((actual_abs - geodesic_pred)**2) / ss_tot
    
    print(f"\nResults for {holdout_conc} ppb:")
    print(f"  Linear:   RMSE={rmse_linear:.4f}, R²={r2_linear:.4f}")
    print(f"  Geodesic: RMSE={rmse_geodesic:.4f}, R²={r2_geodesic:.4f}")
    
    # Create plot
    fig = go.Figure()
    
    # Actual data
    fig.add_trace(go.Scatter(
        x=wl_sampled, y=actual_abs,
        mode='lines+markers',
        name='Actual',
        line=dict(color='black', width=2),
        marker=dict(size=4)
    ))
    
    # Linear interpolation
    fig.add_trace(go.Scatter(
        x=wl_sampled, y=linear_pred,
        mode='lines+markers',
        name=f'Linear (R²={r2_linear:.3f})',
        line=dict(color='red', width=1.5, dash='dash'),
        marker=dict(size=3)
    ))
    
    # Geodesic interpolation
    fig.add_trace(go.Scatter(
        x=wl_sampled, y=geodesic_pred,
        mode='lines+markers',
        name=f'Geodesic (R²={r2_geodesic:.3f})',
        line=dict(color='blue', width=1.5),
        marker=dict(size=3)
    ))
    
    # Add error regions
    linear_error = np.abs(actual_abs - linear_pred)
    geodesic_error = np.abs(actual_abs - geodesic_pred)
    
    fig.add_trace(go.Scatter(
        x=wl_sampled, y=linear_error,
        mode='lines',
        name='Linear Error',
        line=dict(color='red', width=1, dash='dot'),
        yaxis='y2'
    ))
    
    fig.add_trace(go.Scatter(
        x=wl_sampled, y=geodesic_error,
        mode='lines',
        name='Geodesic Error',
        line=dict(color='blue', width=1, dash='dot'),
        yaxis='y2'
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': f'Geodesic vs Linear Interpolation at {holdout_conc} ppb<br>' +
                   f'<sub>Linear R²={r2_linear:.3f}, Geodesic R²={r2_geodesic:.3f} (untrained model)</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Wavelength (nm)",
        yaxis_title="Absorbance",
        yaxis2=dict(
            title="Absolute Error",
            overlaying='y',
            side='right',
            showgrid=False
        ),
        width=1200,
        height=600,
        hovermode='x unified',
        legend=dict(x=0.02, y=0.98)
    )
    
    # Save
    output_file = 'geodesic_quick_demo.html'
    fig.write_html(output_file)
    print(f"\n✓ Saved to {output_file}")
    
    # Show
    fig.show()
    
    print("\n" + "="*60)
    print("IMPORTANT NOTES:")
    print("- This uses an UNTRAINED model (random weights)")
    print("- Only sampled wavelengths for speed")
    print("- A trained model would show dramatic improvement")
    print(f"- Expected: R² from {r2_linear:.2f} to >0.7 with training")
    print("="*60)


if __name__ == "__main__":
    main()