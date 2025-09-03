"""
3D visualization for geodesic vs basic interpolation comparison
Creates interactive Plotly plots
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
from typing import Dict, Optional


def create_3d_comparison_plot(
    predictions_csv: str,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> go.Figure:
    """
    Create 3D surface plots comparing Basic vs Geodesic methods
    
    Args:
        predictions_csv: Path to predictions CSV file
        save_path: Optional path to save HTML file
        show_plot: Whether to display the plot
        
    Returns:
        Plotly figure object
    """
    # Load predictions
    df = pd.read_csv(predictions_csv)
    
    wavelengths = np.sort(df['Wavelength_nm'].unique())
    concentrations = np.sort(df['Concentration_ppb'].unique())
    
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
    c_min, c_max = concentrations.min(), concentrations.max()
    c_range = np.linspace(c_min, c_max, 25)
    
    # Downsample wavelengths for cleaner visualization
    step = max(1, len(wavelengths) // 30)
    wl_subset = wavelengths[::step]
    
    # Create meshgrid
    C_mesh, WL_mesh = np.meshgrid(c_range, wl_subset)
    
    for idx, holdout_conc in enumerate(concentrations):
        # Get data for this holdout
        holdout_data = df[df['Concentration_ppb'] == holdout_conc]
        
        # Sort by wavelength
        holdout_data = holdout_data.sort_values('Wavelength_nm')
        
        wl = holdout_data['Wavelength_nm'].values
        actual = holdout_data['Actual'].values
        basic = holdout_data['BasicInterp'].values
        geodesic = holdout_data['Geodesic'].values
        
        # Determine subplot position
        row = idx // 2 + 1
        col_base = 1 if (idx % 2 == 0) else 3
        
        # Create surfaces by extending the predictions across concentration range
        # For visualization, we'll show how the prediction varies
        basic_surface = np.zeros_like(C_mesh)
        geodesic_surface = np.zeros_like(C_mesh)
        
        for j, wl_val in enumerate(wl_subset):
            wl_idx = np.argmin(np.abs(wl - wl_val))
            # Create a simple surface showing the prediction
            basic_surface[j, :] = basic[wl_idx] * np.ones_like(c_range)
            geodesic_surface[j, :] = geodesic[wl_idx] * np.ones_like(c_range)
        
        # Subsample actual data for markers
        actual_subset = actual[::step]
        c_actual = np.full(len(wl_subset), holdout_conc)
        
        # Add Basic surface
        fig.add_trace(
            go.Surface(
                x=C_mesh, y=WL_mesh, z=basic_surface,
                colorscale='Reds',
                opacity=0.8,
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
                opacity=0.8,
                showscale=False,
                name=f'Geodesic {holdout_conc:.0f} ppb'
            ),
            row=row, col=col_base + 1
        )
        
        # Add actual data points as markers on both surfaces
        for col_offset in [0, 1]:
            fig.add_trace(
                go.Scatter3d(
                    x=c_actual,
                    y=wl_subset,
                    z=actual_subset,
                    mode='markers',
                    marker=dict(size=3, color='black'),
                    showlegend=False,
                    name='Actual'
                ),
                row=row, col=col_base + col_offset
            )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Geodesic vs Basic Interpolation - All Holdouts',
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


def create_single_holdout_3d(
    predictions_csv: str,
    holdout_conc: float,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> go.Figure:
    """
    Create detailed 3D plot for a single holdout concentration
    
    Args:
        predictions_csv: Path to predictions CSV
        holdout_conc: Concentration to visualize
        save_path: Optional path to save HTML
        show_plot: Whether to display the plot
        
    Returns:
        Plotly figure object
    """
    # Load data
    df = pd.read_csv(predictions_csv)
    
    # Filter for specific holdout
    holdout_data = df[df['Concentration_ppb'] == holdout_conc]
    holdout_data = holdout_data.sort_values('Wavelength_nm')
    
    wl = holdout_data['Wavelength_nm'].values
    actual = holdout_data['Actual'].values
    basic = holdout_data['BasicInterp'].values
    geodesic = holdout_data['Geodesic'].values
    
    # Create figure with 1 row, 2 columns
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=[f'Basic Interpolation - {holdout_conc:.0f} ppb',
                       f'Geodesic Model - {holdout_conc:.0f} ppb'],
        horizontal_spacing=0.1
    )
    
    # Create a simple concentration range around the holdout
    c_range = np.linspace(holdout_conc - 10, holdout_conc + 10, 20)
    C_mesh, WL_mesh = np.meshgrid(c_range, wl)
    
    # Create surfaces
    basic_surface = np.tile(basic[:, np.newaxis], (1, len(c_range)))
    geodesic_surface = np.tile(geodesic[:, np.newaxis], (1, len(c_range)))
    
    # Add surfaces
    fig.add_trace(
        go.Surface(
            x=C_mesh, y=WL_mesh, z=basic_surface,
            colorscale='Reds',
            opacity=0.8,
            showscale=True,
            name='Basic'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Surface(
            x=C_mesh, y=WL_mesh, z=geodesic_surface,
            colorscale='Blues',
            opacity=0.8,
            showscale=True,
            name='Geodesic'
        ),
        row=1, col=2
    )
    
    # Add actual data as markers
    c_actual = np.full_like(wl, holdout_conc)
    
    for col in [1, 2]:
        fig.add_trace(
            go.Scatter3d(
                x=c_actual, y=wl, z=actual,
                mode='markers',
                marker=dict(size=2, color='black'),
                showlegend=False,
                name='Actual'
            ),
            row=1, col=col
        )
    
    # Update layout
    fig.update_layout(
        title={
            'text': f'Detailed Comparison - {holdout_conc:.0f} ppb Holdout',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        width=1400,
        height=600,
        showlegend=False,
        scene=dict(
            xaxis_title="Concentration (ppb)",
            yaxis_title="Wavelength (nm)",
            zaxis_title="Absorbance"
        ),
        scene2=dict(
            xaxis_title="Concentration (ppb)",
            yaxis_title="Wavelength (nm)",
            zaxis_title="Absorbance"
        )
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"💾 Single holdout 3D saved to {save_path}")
    
    if show_plot:
        fig.show()
    
    return fig