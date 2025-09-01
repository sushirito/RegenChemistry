#!/usr/bin/env python3
"""
Interactive 3D Visualization of UV-Vis Spectral Data
Creates a beautiful curved manifold showing concentration-wavelength-absorbance relationships
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

def create_3d_manifold(wavelengths, concentrations, absorbance_matrix):
    """Create interactive 3D surface plot with aesthetic enhancements"""
    
    # Create meshgrid for 3D surface
    X, Y = np.meshgrid(wavelengths, concentrations)
    Z = absorbance_matrix.T  # Transpose to match meshgrid dimensions
    
    # Create the main surface plot
    surface = go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale='Viridis',  # Beautiful color gradient
        showscale=True,
        colorbar=dict(
            title="Absorbance",
            titleside="right",
            tickmode="linear",
            tick0=0,
            dtick=0.05,
            len=0.75,
            thickness=15
        ),
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
        name='Spectral Surface'
    )
    
    # Add scatter points on the surface for actual data points
    # Sample points for visualization (every 50th wavelength to avoid overcrowding)
    sample_indices = range(0, len(wavelengths), 50)
    scatter_points = []
    
    for i in sample_indices:
        for j, conc in enumerate(concentrations):
            scatter_points.append(go.Scatter3d(
                x=[wavelengths[i]],
                y=[conc],
                z=[absorbance_matrix[i, j]],
                mode='markers',
                marker=dict(
                    size=3,
                    color=absorbance_matrix[i, j],
                    colorscale='Plasma',
                    showscale=False,
                    line=dict(color='white', width=0.5)
                ),
                showlegend=False,
                hovertemplate=f'λ: {wavelengths[i]:.0f} nm<br>' +
                             f'Conc: {conc:.0f} ppb<br>' +
                             f'Abs: {absorbance_matrix[i, j]:.4f}<extra></extra>'
            ))
    
    # Create the figure
    fig = go.Figure(data=[surface] + scatter_points)
    
    # Update layout for aesthetics
    fig.update_layout(
        title={
            'text': 'UV-Vis Spectral Response Surface<br><sub>Arsenic Detection System - Concentration vs Wavelength vs Absorbance</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2c3e50'}
        },
        scene=dict(
            xaxis=dict(
                title='Wavelength (nm)',
                titlefont=dict(size=14),
                gridcolor='lightgray',
                showbackground=True,
                backgroundcolor='rgba(230, 230, 250, 0.1)',
                range=[200, 800]
            ),
            yaxis=dict(
                title='Concentration (ppb)',
                titlefont=dict(size=14),
                gridcolor='lightgray',
                showbackground=True,
                backgroundcolor='rgba(230, 250, 230, 0.1)',
                range=[0, 60]
            ),
            zaxis=dict(
                title='Absorbance',
                titlefont=dict(size=14),
                gridcolor='lightgray',
                showbackground=True,
                backgroundcolor='rgba(250, 230, 230, 0.1)'
            ),
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.2),
                center=dict(x=0, y=0, z=0)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1.5, y=1, z=0.7)
        ),
        width=1200,
        height=800,
        margin=dict(l=0, r=0, t=80, b=0),
        paper_bgcolor='#f8f9fa',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    
    # Add animation frames for rotation effect (optional)
    def create_rotation_frames():
        frames = []
        for angle in range(0, 360, 10):
            rad = np.radians(angle)
            frames.append(go.Frame(
                layout=dict(
                    scene_camera_eye=dict(
                        x=2 * np.cos(rad),
                        y=2 * np.sin(rad),
                        z=1.5
                    )
                )
            ))
        return frames
    
    # Add play/pause button for rotation
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=0.95,
                x=0.02,
                xanchor="left",
                yanchor="top",
                buttons=[
                    dict(
                        label="Rotate",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": 100, "redraw": True},
                            "fromcurrent": True,
                            "mode": "immediate"
                        }]
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }]
                    )
                ]
            )
        ]
    )
    
    # Add rotation frames
    fig.frames = create_rotation_frames()
    
    return fig

def add_cross_sections(fig, wavelengths, concentrations, absorbance_matrix):
    """Add interactive cross-section views at specific wavelengths/concentrations"""
    
    # Add cross-section at 520nm (typical peak wavelength)
    target_wavelength = 520
    idx = np.argmin(np.abs(wavelengths - target_wavelength))
    
    cross_section = go.Scatter3d(
        x=[wavelengths[idx]] * len(concentrations),
        y=concentrations,
        z=absorbance_matrix[idx, :],
        mode='lines+markers',
        line=dict(color='red', width=4),
        marker=dict(size=5, color='darkred'),
        name=f'Cross-section at {wavelengths[idx]:.0f} nm',
        hovertemplate='Conc: %{y:.0f} ppb<br>Abs: %{z:.4f}<extra></extra>'
    )
    
    fig.add_trace(cross_section)
    
    # Add cross-section at 30 ppb concentration
    target_conc_idx = 3  # 30 ppb
    conc_cross = go.Scatter3d(
        x=wavelengths,
        y=[concentrations[target_conc_idx]] * len(wavelengths),
        z=absorbance_matrix[:, target_conc_idx],
        mode='lines',
        line=dict(color='blue', width=3),
        name=f'Cross-section at {concentrations[target_conc_idx]:.0f} ppb',
        hovertemplate='λ: %{x:.0f} nm<br>Abs: %{z:.4f}<extra></extra>'
    )
    
    fig.add_trace(conc_cross)
    
    return fig

def main():
    """Main function to create and save the visualization"""
    
    # Load data
    filepath = '/Users/aditya/CodingProjects/STS_September/0.30MB_AuNP_As.csv'
    print("Loading spectral data...")
    wavelengths, concentrations, absorbance_matrix = load_spectral_data(filepath)
    
    print(f"Data shape: {len(wavelengths)} wavelengths × {len(concentrations)} concentrations")
    print(f"Wavelength range: {wavelengths.min():.0f} - {wavelengths.max():.0f} nm")
    print(f"Concentration range: {min(concentrations):.0f} - {max(concentrations):.0f} ppb")
    
    # Create 3D visualization
    print("Creating 3D manifold visualization...")
    fig = create_3d_manifold(wavelengths, concentrations, absorbance_matrix)
    
    # Add cross-sections for better understanding
    fig = add_cross_sections(fig, wavelengths, concentrations, absorbance_matrix)
    
    # Save as interactive HTML
    output_file = 'spectral_manifold.html'
    print(f"Saving interactive visualization to {output_file}...")
    fig.write_html(
        output_file,
        config={'displayModeBar': True, 'displaylogo': False},
        include_plotlyjs='cdn'
    )
    
    # Also show in browser
    fig.show()
    
    print(f"✓ Visualization complete! Open '{output_file}' in your browser to interact with the 3D plot.")
    print("\nInteractive features:")
    print("  • Click and drag to rotate the surface")
    print("  • Scroll to zoom in/out")
    print("  • Hover over points to see exact values")
    print("  • Click 'Rotate' button for automatic rotation")
    print("  • Use legend to toggle cross-sections on/off")

if __name__ == "__main__":
    main()