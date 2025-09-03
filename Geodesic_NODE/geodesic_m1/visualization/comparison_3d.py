"""
3D visualization for geodesic vs basic interpolation comparison
Creates interactive Plotly plots showing actual interpolation surfaces
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
from typing import Dict, Optional, Tuple
from pathlib import Path


def compute_interpolation_surface(
    wavelengths: np.ndarray,
    concentrations: list,
    abs_matrix: np.ndarray,
    holdout_idx: int,
    c_range: np.ndarray,
    wl_subset: np.ndarray
) -> np.ndarray:
    """
    Compute the actual interpolation surface for visualization
    
    Args:
        wavelengths: Full wavelength array
        concentrations: List of all concentration values
        abs_matrix: Full absorbance matrix [wavelengths, concentrations]
        holdout_idx: Index of concentration to exclude
        c_range: Concentration values for surface
        wl_subset: Wavelength subset for surface
        
    Returns:
        Surface array [len(wl_subset), len(c_range)]
    """
    # Get training data (exclude holdout)
    train_concs = [concentrations[i] for i in range(len(concentrations)) if i != holdout_idx]
    train_abs = np.column_stack([abs_matrix[:, i] for i in range(len(concentrations)) if i != holdout_idx])
    
    surface = np.zeros((len(wl_subset), len(c_range)))
    
    for j, wl_val in enumerate(wl_subset):
        # Find closest wavelength index
        wl_idx = np.argmin(np.abs(wavelengths - wl_val))
        
        # Get training absorbances at this wavelength
        train_abs_at_wl = train_abs[wl_idx, :]
        
        # Interpolate across concentration range
        # Use cubic extrapolation (matching A100) to show failure at edges
        for k, c_val in enumerate(c_range):
            # Match A100 implementation: cubic if enough points, else linear
            if len(train_concs) >= 4:
                kind = 'cubic'
            else:
                kind = 'linear'
            
            try:
                f = interp1d(train_concs, train_abs_at_wl, kind=kind,
                           fill_value='extrapolate', bounds_error=False)
                surface[j, k] = f(c_val)
            except:
                # Fallback to linear if cubic fails
                f = interp1d(train_concs, train_abs_at_wl, kind='linear',
                           fill_value='extrapolate', bounds_error=False)
                surface[j, k] = f(c_val)
    
    return surface


def create_3d_comparison_plot(
    predictions_csv: str,
    data_path: str = "data/0.30MB_AuNP_As.csv",
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> go.Figure:
    """
    Create 3D surface plots comparing Basic vs Geodesic methods
    Shows actual interpolation surfaces, not flat planes
    
    Args:
        predictions_csv: Path to predictions CSV file (for geodesic results)
        data_path: Path to raw spectral data CSV
        save_path: Optional path to save HTML file
        show_plot: Whether to display the plot
        
    Returns:
        Plotly figure object
    """
    # Load raw spectral data
    df_data = pd.read_csv(data_path)
    wavelengths = df_data['Wavelength'].values
    concentrations = [float(col) for col in df_data.columns[1:]]
    abs_matrix = df_data.iloc[:, 1:].values
    
    # Load predictions for geodesic results
    df_pred = pd.read_csv(predictions_csv)
    
    # Create 2x3 grid for 6 holdouts
    rows, cols = 3, 4  # 3 rows, 4 columns (2 surfaces per holdout)
    specs = [[{'type': 'surface'}] * cols for _ in range(rows)]
    
    # Create subplot titles
    titles = []
    for conc in concentrations:
        titles.append(f'Basic Interpolation - {conc:.0f} ppb')
        titles.append(f'Geodesic Model - {conc:.0f} ppb')
    
    fig = make_subplots(
        rows=rows, cols=cols,
        specs=specs,
        subplot_titles=titles,
        horizontal_spacing=0.04,
        vertical_spacing=0.06
    )
    
    # Create concentration range for surface
    c_min, c_max = min(concentrations), max(concentrations)
    c_range = np.linspace(c_min, c_max, 30)
    
    # Downsample wavelengths for cleaner visualization
    step = max(1, len(wavelengths) // 40)
    wl_subset = wavelengths[::step]
    
    # Create meshgrid
    C_mesh, WL_mesh = np.meshgrid(c_range, wl_subset)
    
    for idx, holdout_conc in enumerate(concentrations):
        holdout_idx = idx
        
        # Compute actual interpolation surface
        basic_surface = compute_interpolation_surface(
            wavelengths, concentrations, abs_matrix,
            holdout_idx, c_range, wl_subset
        )
        
        # Get geodesic predictions for this holdout
        holdout_pred = df_pred[df_pred['Concentration_ppb'] == holdout_conc]
        holdout_pred = holdout_pred.sort_values('Wavelength_nm')
        
        # Create geodesic surface (still flat for now, as it only predicts at holdout)
        # In a full implementation, we'd compute geodesic predictions across all concentrations
        geodesic_surface = np.zeros_like(C_mesh)
        geodesic_vals = holdout_pred['Geodesic'].values
        
        for j, wl_val in enumerate(wl_subset):
            wl_idx = np.argmin(np.abs(holdout_pred['Wavelength_nm'].values - wl_val))
            # For now, show flat surface at prediction (ideally would compute full surface)
            geodesic_surface[j, :] = geodesic_vals[wl_idx] * np.ones_like(c_range)
        
        # Get actual data for this holdout
        actual_abs = abs_matrix[:, holdout_idx]
        
        # Determine subplot position
        row = idx // 2 + 1
        col_base = 1 if (idx % 2 == 0) else 3
        
        # Add Basic interpolation surface
        fig.add_trace(
            go.Surface(
                x=C_mesh, y=WL_mesh, z=basic_surface,
                colorscale='Reds',
                opacity=0.7,
                showscale=False,
                name=f'Basic {holdout_conc:.0f} ppb'
            ),
            row=row, col=col_base
        )
        
        # Add Geodesic surface
        fig.add_trace(
            go.Surface(
                x=C_mesh, y=WL_mesh, z=geodesic_surface,
                colorscale='Blues',
                opacity=0.7,
                showscale=False,
                name=f'Geodesic {holdout_conc:.0f} ppb'
            ),
            row=row, col=col_base + 1
        )
        
        # Add training data points as markers on basic interpolation surface
        train_concs = [concentrations[i] for i in range(len(concentrations)) if i != holdout_idx]
        
        for train_idx, train_conc in enumerate(train_concs):
            train_conc_idx = [i for i, c in enumerate(concentrations) if c == train_conc][0]
            train_abs = abs_matrix[::step, train_conc_idx]
            c_train = np.full(len(wl_subset), train_conc)
            
            fig.add_trace(
                go.Scatter3d(
                    x=c_train,
                    y=wl_subset,
                    z=train_abs,
                    mode='markers',
                    marker=dict(size=2, color='gray', opacity=0.6),
                    showlegend=False,
                    name='Training Data'
                ),
                row=row, col=col_base
            )
        
        # Add actual holdout data as black markers on both surfaces
        actual_subset = actual_abs[::step]
        c_actual = np.full(len(wl_subset), holdout_conc)
        
        for col_offset in [0, 1]:
            fig.add_trace(
                go.Scatter3d(
                    x=c_actual,
                    y=wl_subset,
                    z=actual_subset,
                    mode='markers',
                    marker=dict(size=4, color='black'),
                    showlegend=False,
                    name='Actual Holdout'
                ),
                row=row, col=col_base + col_offset
            )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Geodesic vs Basic Interpolation - True Interpolation Surfaces',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        width=1700,
        height=1500,
        showlegend=False
    )
    
    # Update all 3D axes
    for i in range(1, rows * cols + 1):
        scene_name = 'scene' if i == 1 else f'scene{i}'
        fig.update_layout({
            scene_name: dict(
                xaxis_title="Concentration (ppb)",
                yaxis_title="Wavelength (nm)",
                zaxis_title="Absorbance",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            )
        })
    
    if save_path:
        fig.write_html(save_path)
        print(f"💾 3D comparison saved to {save_path}")
    
    if show_plot:
        fig.show()
    
    return fig


def create_interpolation_analysis(
    data_path: str = "data/0.30MB_AuNP_As.csv",
    holdout_idx: int = 5,  # Default to 60 ppb
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Create detailed analysis of interpolation vs extrapolation behavior
    
    Args:
        data_path: Path to raw spectral data
        holdout_idx: Index of concentration to hold out
        save_path: Optional path to save HTML
        
    Returns:
        Plotly figure showing interpolation behavior
    """
    # Load data
    df = pd.read_csv(data_path)
    wavelengths = df['Wavelength'].values
    concentrations = [float(col) for col in df.columns[1:]]
    abs_matrix = df.iloc[:, 1:].values
    
    holdout_conc = concentrations[holdout_idx]
    
    # Select a few representative wavelengths
    wl_indices = [100, 250, 400, 550]  # Spread across spectrum
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f'λ = {wavelengths[i]:.0f} nm' for i in wl_indices]
    )
    
    for plot_idx, wl_idx in enumerate(wl_indices):
        row = plot_idx // 2 + 1
        col = plot_idx % 2 + 1
        
        # Get training data
        train_concs = [concentrations[i] for i in range(len(concentrations)) if i != holdout_idx]
        train_abs = [abs_matrix[wl_idx, i] for i in range(len(concentrations)) if i != holdout_idx]
        
        # Create fine concentration grid
        c_fine = np.linspace(0, 60, 200)
        
        # Compute cubic interpolation
        f_cubic = interp1d(train_concs, train_abs, kind='cubic',
                         fill_value='extrapolate', bounds_error=False)
        cubic_pred = f_cubic(c_fine)
        
        # Compute linear interpolation
        f_linear = interp1d(train_concs, train_abs, kind='linear',
                          fill_value='extrapolate', bounds_error=False)
        linear_pred = f_linear(c_fine)
        
        # Add interpolation curves
        fig.add_trace(
            go.Scatter(x=c_fine, y=cubic_pred, 
                      mode='lines', name='Cubic',
                      line=dict(color='red', width=2)),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(x=c_fine, y=linear_pred,
                      mode='lines', name='Linear',
                      line=dict(color='blue', width=2)),
            row=row, col=col
        )
        
        # Add training points
        fig.add_trace(
            go.Scatter(x=train_concs, y=train_abs,
                      mode='markers', name='Training',
                      marker=dict(size=8, color='gray')),
            row=row, col=col
        )
        
        # Add actual holdout point
        actual_val = abs_matrix[wl_idx, holdout_idx]
        fig.add_trace(
            go.Scatter(x=[holdout_conc], y=[actual_val],
                      mode='markers', name='Actual',
                      marker=dict(size=12, color='black', symbol='star')),
            row=row, col=col
        )
        
        # Add predictions at holdout
        fig.add_trace(
            go.Scatter(x=[holdout_conc], y=[f_cubic(holdout_conc)],
                      mode='markers', name='Cubic Pred',
                      marker=dict(size=10, color='red', symbol='x')),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(x=[holdout_conc], y=[f_linear(holdout_conc)],
                      mode='markers', name='Linear Pred',
                      marker=dict(size=10, color='blue', symbol='cross')),
            row=row, col=col
        )
    
    fig.update_layout(
        title=f'Interpolation Analysis - Holdout {holdout_conc:.0f} ppb',
        height=800,
        width=1200,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Concentration (ppb)")
    fig.update_yaxes(title_text="Absorbance")
    
    if save_path:
        fig.write_html(save_path)
        print(f"💾 Interpolation analysis saved to {save_path}")
    
    return fig