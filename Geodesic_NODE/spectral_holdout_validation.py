#!/usr/bin/env python3
"""
Leave-One-Out Validation Visualization for UV-Vis Spectral Data
Creates 6 beautiful 3D manifolds showing interpolation validation for each held-out concentration
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d, griddata

def load_spectral_data(filepath):
    """Load and reshape spectral data for 3D visualization"""
    # Read CSV file
    df = pd.read_csv(filepath)
    
    # Extract wavelengths (first column)
    wavelengths = df['Wavelength'].values
    
    # Extract concentration values from column headers
    concentrations = [float(col) for col in df.columns[1:]]
    
    # Create absorbance matrix (wavelengths x concentrations)
    absorbance_matrix = df.iloc[:, 1:].values
    
    return wavelengths, concentrations, absorbance_matrix

def interpolate_holdout_concentration(wavelengths, concentrations, absorbance_matrix, holdout_idx):
    """Interpolate the held-out concentration using remaining data"""
    
    # Create training data (excluding holdout)
    train_concs = concentrations[:holdout_idx] + concentrations[holdout_idx+1:]
    train_abs = np.column_stack([absorbance_matrix[:, i] for i in range(len(concentrations)) if i != holdout_idx])
    
    # Interpolate for each wavelength
    interpolated_abs = np.zeros(len(wavelengths))
    holdout_conc = concentrations[holdout_idx]
    
    for i, wl in enumerate(wavelengths):
        # Use cubic interpolation if enough points, otherwise linear
        if len(train_concs) > 3:
            interp_func = interp1d(train_concs, train_abs[i, :], kind='cubic', 
                                  fill_value='extrapolate', bounds_error=False)
        else:
            interp_func = interp1d(train_concs, train_abs[i, :], kind='linear',
                                  fill_value='extrapolate', bounds_error=False)
        interpolated_abs[i] = interp_func(holdout_conc)
    
    return interpolated_abs

def create_holdout_surface(wavelengths, concentrations, absorbance_matrix, holdout_idx, 
                          row, col, fig, show_colorbar=False):
    """Create a single 3D surface plot for leave-one-out validation"""
    
    # Get interpolated values for the holdout concentration
    interpolated_abs = interpolate_holdout_concentration(wavelengths, concentrations, 
                                                        absorbance_matrix, holdout_idx)
    
    # Create extended surface that includes interpolated values at holdout position
    # Build the full concentration range surface by combining training + interpolated data
    extended_absorbance = np.zeros((len(wavelengths), len(concentrations)))
    
    for j, conc in enumerate(concentrations):
        if j == holdout_idx:
            # Use interpolated values for the holdout concentration
            extended_absorbance[:, j] = interpolated_abs
        else:
            # Use actual values for training concentrations
            extended_absorbance[:, j] = absorbance_matrix[:, j]
    
    # Create meshgrid for full concentration range (0-60)
    X_full, Y_full = np.meshgrid(wavelengths, concentrations)
    Z_full = extended_absorbance.T
    
    # Create the extended surface
    surface = go.Surface(
        x=X_full,
        y=Y_full,
        z=Z_full,
        colorscale='Viridis',
        showscale=show_colorbar,
        colorbar=dict(
            title="Absorbance",
            titleside="right",
            tickmode="linear",
            tick0=0,
            dtick=0.05,
            len=0.75,
            thickness=15,
            x=1.02
        ) if show_colorbar else None,
        contours={
            "z": {"show": True, "usecolormap": True, "highlightcolor": "limegreen", "project": {"z": True}},
            "x": {"show": True, "usecolormap": False, "color": "white", "width": 1, "highlight": False},
            "y": {"show": True, "usecolormap": False, "color": "white", "width": 1, "highlight": False}
        },
        opacity=0.95,
        lightposition=dict(x=-45, y=45, z=50),
        lighting=dict(
            ambient=0.4,
            diffuse=0.5,
            specular=0.2,
            roughness=0.5,
            fresnel=0.2
        ),
        name='Interpolated Surface',
        showlegend=False,
        hovertemplate='λ: %{x:.0f} nm<br>Conc: %{y:.0f} ppb<br>Abs: %{z:.4f}<extra></extra>'
    )
    
    # Add actual holdout data as red line
    holdout_conc = concentrations[holdout_idx]
    actual_line = go.Scatter3d(
        x=wavelengths,
        y=[holdout_conc] * len(wavelengths),
        z=absorbance_matrix[:, holdout_idx],
        mode='lines',
        line=dict(color='red', width=4),
        showlegend=False,
        hovertemplate='λ: %{x:.0f} nm<br>Actual Abs: %{z:.4f}<extra></extra>'
    )
    
    # Add traces to subplot
    fig.add_trace(surface, row=row, col=col)
    fig.add_trace(actual_line, row=row, col=col)
    
    # Calculate RMSE for this holdout
    rmse = np.sqrt(np.mean((absorbance_matrix[:, holdout_idx] - interpolated_abs)**2))
    
    # Update subplot title
    fig.update_scenes(
        dict(
            xaxis=dict(
                title='Wavelength (nm)',
                titlefont=dict(size=10),
                gridcolor='lightgray',
                showbackground=True,
                backgroundcolor='rgba(230, 230, 250, 0.1)',
                range=[200, 800]
            ),
            yaxis=dict(
                title='Concentration (ppb)',
                titlefont=dict(size=10),
                gridcolor='lightgray',
                showbackground=True,
                backgroundcolor='rgba(230, 250, 230, 0.1)',
                range=[0, 60]
            ),
            zaxis=dict(
                title='Absorbance',
                titlefont=dict(size=10),
                gridcolor='lightgray',
                showbackground=True,
                backgroundcolor='rgba(250, 230, 230, 0.1)'
            ),
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.2),
                center=dict(x=0, y=0, z=0)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1.5, y=1, z=0.7),
            annotations=[
                dict(
                    showarrow=False,
                    x=0.5,
                    y=1.1,
                    z=0,
                    text=f'<b>Holdout: {holdout_conc:.0f} ppb</b><br>RMSE: {rmse:.4f}',
                    xanchor='center',
                    yanchor='top',
                    font=dict(size=12, color='#2c3e50')
                )
            ]
        ),
        row=row, col=col
    )

def main():
    """Main function to create and save the visualization"""
    
    # Load data
    filepath = '/Users/aditya/CodingProjects/STS_September/0.30MB_AuNP_As.csv'
    print("Loading spectral data...")
    wavelengths, concentrations, absorbance_matrix = load_spectral_data(filepath)
    
    print(f"Data shape: {len(wavelengths)} wavelengths × {len(concentrations)} concentrations")
    print(f"Creating leave-one-out validation for concentrations: {concentrations}")
    
    # Create subplot figure with 2 rows, 3 columns
    fig = make_subplots(
        rows=2, cols=3,
        specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}],
               [{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=[f'Holdout: {conc:.0f} ppb' for conc in concentrations],
        horizontal_spacing=0.08,
        vertical_spacing=0.12
    )
    
    # Create 6 holdout validation plots
    positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
    
    for idx, (row, col) in enumerate(positions):
        if idx < len(concentrations):
            print(f"Creating validation plot for holdout concentration: {concentrations[idx]:.0f} ppb")
            # Only show colorbar for the last plot
            show_colorbar = (idx == len(concentrations) - 1)
            create_holdout_surface(wavelengths, concentrations, absorbance_matrix, 
                                 idx, row, col, fig, show_colorbar)
    
    # Update overall layout
    fig.update_layout(
        title={
            'text': 'Leave-One-Out Validation: UV-Vis Spectral Interpolation<br><sub>Arsenic Detection System - Testing Interpolation Accuracy at Each Concentration</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2c3e50'},
            'y': 0.98
        },
        width=1800,
        height=1000,
        margin=dict(l=0, r=50, t=100, b=0),
        paper_bgcolor='#f8f9fa',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        ),
        showlegend=False
    )
    
    # Save as interactive HTML
    output_file = 'spectral_holdout_validation.html'
    print(f"Saving interactive visualization to {output_file}...")
    fig.write_html(
        output_file,
        config={'displayModeBar': True, 'displaylogo': False},
        include_plotlyjs='cdn'
    )
    
    # Also show in browser
    fig.show()
    
    # Calculate overall statistics
    print("\n" + "="*60)
    print("VALIDATION RESULTS SUMMARY")
    print("="*60)
    
    total_rmse = 0
    for idx, conc in enumerate(concentrations):
        interpolated_abs = interpolate_holdout_concentration(wavelengths, concentrations, 
                                                           absorbance_matrix, idx)
        rmse = np.sqrt(np.mean((absorbance_matrix[:, idx] - interpolated_abs)**2))
        max_error = np.max(np.abs(absorbance_matrix[:, idx] - interpolated_abs))
        print(f"Concentration {conc:5.0f} ppb: RMSE = {rmse:.4f}, Max Error = {max_error:.4f}")
        total_rmse += rmse
    
    print(f"\nAverage RMSE across all concentrations: {total_rmse/len(concentrations):.4f}")
    print("="*60)
    
    print(f"\n✓ Validation complete! Open '{output_file}' in your browser to interact with the 3D plots.")
    print("\nInteractive features:")
    print("  • Click and drag to rotate each surface")
    print("  • Scroll to zoom in/out")
    print("  • Hover over points to see exact values")
    print("  • Red lines show actual held-out data")
    print("  • Surface shows interpolated manifold extending 0-60 ppb")

if __name__ == "__main__":
    main()