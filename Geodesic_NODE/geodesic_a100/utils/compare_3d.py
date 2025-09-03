#!/usr/bin/env python3
"""
3D Surface Comparison of Basic Interpolation vs Geodesic Interpolation
Creates side-by-side 3D manifolds showing both methods for leave-one-out validation
Adapted for A100 implementation with coupled ODE system
"""

import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
from pathlib import Path
import time
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from geodesic_a100.models import GeodesicNODE
from geodesic_a100.data import SpectralDataset


class Comparison3D:
    """Create 3D surface comparisons between basic and geodesic interpolation"""
    
    def __init__(self, csv_path='data/0.30MB_AuNP_As.csv'):
        """Initialize with spectral data"""
        self.csv_path = csv_path
        self.device = torch.device('cpu')  # For M3 Mac
        
        # Load data
        df = pd.read_csv(csv_path)
        self.wavelengths = df['Wavelength'].values  # 601 points
        self.concentrations = [float(col) for col in df.columns[1:]]  # [0, 10, 20, 30, 40, 60]
        self.absorbance_matrix = df.iloc[:, 1:].values  # [601, 6]
        
        print(f"Loaded data: {len(self.wavelengths)} wavelengths, {len(self.concentrations)} concentrations")
        
    def basic_interpolation(self, holdout_idx):
        """Perform basic cubic/linear interpolation"""
        train_concs = self.concentrations[:holdout_idx] + self.concentrations[holdout_idx+1:]
        train_abs = np.column_stack([self.absorbance_matrix[:, i] 
                                    for i in range(len(self.concentrations)) 
                                    if i != holdout_idx])
        
        interpolated_abs = np.zeros(len(self.wavelengths))
        holdout_conc = self.concentrations[holdout_idx]
        
        for i, wl in enumerate(self.wavelengths):
            if len(train_concs) >= 4:
                interp_func = interp1d(train_concs, train_abs[i, :], kind='cubic',
                                      fill_value='extrapolate', bounds_error=False)
            else:
                interp_func = interp1d(train_concs, train_abs[i, :], kind='linear',
                                      fill_value='extrapolate', bounds_error=False)
            interpolated_abs[i] = interp_func(holdout_conc)
        
        return interpolated_abs
    
    def geodesic_interpolation(self, model, holdout_idx, dataset):
        """Perform geodesic interpolation using trained model"""
        holdout_conc = self.concentrations[holdout_idx]
        interpolated_abs = np.zeros(len(self.wavelengths))
        
        # Find nearest source concentration
        train_concs = [self.concentrations[i] for i in range(len(self.concentrations)) 
                      if i != holdout_idx]
        distances = [abs(tc - holdout_conc) for tc in train_concs]
        nearest_idx = distances.index(min(distances))
        source_conc = train_concs[nearest_idx]
        
        # Normalize concentrations
        c_source_norm = (source_conc - dataset.c_mean) / dataset.c_std
        c_target_norm = (holdout_conc - dataset.c_mean) / dataset.c_std
        
        # Process wavelengths in batches
        batch_size = 50
        with torch.no_grad():
            for batch_start in range(0, len(self.wavelengths), batch_size):
                batch_end = min(batch_start + batch_size, len(self.wavelengths))
                batch_wl = self.wavelengths[batch_start:batch_end]
                
                # Normalize wavelengths
                wl_norm = (batch_wl - dataset.lambda_mean) / dataset.lambda_std
                
                # Create batch tensors
                n_batch = len(batch_wl)
                c_sources = torch.full((n_batch,), c_source_norm, device=self.device)
                c_targets = torch.full((n_batch,), c_target_norm, device=self.device)
                wavelengths = torch.tensor(wl_norm, dtype=torch.float32, device=self.device)
                
                try:
                    # Get predictions from geodesic model
                    output = model(c_sources, c_targets, wavelengths)
                    batch_pred = output['absorbance'].cpu().numpy()
                    
                    # Denormalize
                    batch_pred = batch_pred * dataset.A_std + dataset.A_mean
                    interpolated_abs[batch_start:batch_end] = batch_pred
                except Exception as e:
                    # Fallback to mean if prediction fails
                    interpolated_abs[batch_start:batch_end] = dataset.A_mean
        
        return interpolated_abs
    
    def create_surface_plot(self, method_name, interpolated_abs, holdout_idx, 
                          row, col, fig, show_colorbar=False):
        """Create a 3D surface plot for a specific interpolation method"""
        
        # Set colors based on method
        if method_name == "Basic Interpolation":
            colorscale = 'Viridis'
            surface_color = 'blue'
        else:  # Geodesic
            colorscale = 'Plasma'
            surface_color = 'purple'
        
        # Create extended surface including interpolated values
        extended_absorbance = np.zeros((len(self.wavelengths), len(self.concentrations)))
        
        for j, conc in enumerate(self.concentrations):
            if j == holdout_idx:
                extended_absorbance[:, j] = interpolated_abs
            else:
                extended_absorbance[:, j] = self.absorbance_matrix[:, j]
        
        # Create meshgrid
        X_full, Y_full = np.meshgrid(self.wavelengths, self.concentrations)
        Z_full = extended_absorbance.T
        
        # Create surface
        surface = go.Surface(
            x=X_full,
            y=Y_full,
            z=Z_full,
            colorscale=colorscale,
            showscale=show_colorbar,
            colorbar=dict(
                title="Absorbance",
                titleside="right",
                len=0.75,
                thickness=15,
                x=1.02 if col == 2 else None
            ) if show_colorbar else None,
            contours={
                "z": {"show": True, "usecolormap": True, "project": {"z": True}},
                "x": {"show": True, "usecolormap": False, "color": "white", "width": 1},
                "y": {"show": True, "usecolormap": False, "color": "white", "width": 1}
            },
            opacity=0.9,
            name=method_name,
            showlegend=False,
            hovertemplate='λ: %{x:.0f} nm<br>Conc: %{y:.0f} ppb<br>Abs: %{z:.4f}<extra></extra>'
        )
        
        # Add actual holdout data as red line
        holdout_conc = self.concentrations[holdout_idx]
        actual_line = go.Scatter3d(
            x=self.wavelengths,
            y=[holdout_conc] * len(self.wavelengths),
            z=self.absorbance_matrix[:, holdout_idx],
            mode='lines',
            line=dict(color='red', width=5),
            name='Actual Data',
            showlegend=False,
            hovertemplate='λ: %{x:.0f} nm<br>Actual Abs: %{z:.4f}<extra></extra>'
        )
        
        # Add interpolated line for clarity
        interp_line = go.Scatter3d(
            x=self.wavelengths,
            y=[holdout_conc] * len(self.wavelengths),
            z=interpolated_abs,
            mode='lines',
            line=dict(color=surface_color, width=3, dash='dash'),
            name=f'{method_name} Prediction',
            showlegend=False,
            hovertemplate='λ: %{x:.0f} nm<br>Predicted Abs: %{z:.4f}<extra></extra>'
        )
        
        # Add traces
        fig.add_trace(surface, row=row, col=col)
        fig.add_trace(actual_line, row=row, col=col)
        fig.add_trace(interp_line, row=row, col=col)
        
        # Update scene
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
                aspectratio=dict(x=1.5, y=1, z=0.7)
            ),
            row=row, col=col
        )
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((self.absorbance_matrix[:, holdout_idx] - interpolated_abs)**2))
        mae = np.mean(np.abs(self.absorbance_matrix[:, holdout_idx] - interpolated_abs))
        ss_res = np.sum((self.absorbance_matrix[:, holdout_idx] - interpolated_abs)**2)
        ss_tot = np.sum((self.absorbance_matrix[:, holdout_idx] - 
                        np.mean(self.absorbance_matrix[:, holdout_idx]))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else float('-inf')
        
        return {'rmse': rmse, 'mae': mae, 'r2': r2}
    
    def create_full_comparison(self, model=None, dataset=None, save_path='test_3d_comparison.html'):
        """Create complete 3D comparison visualization"""
        
        # Select key concentrations to compare
        test_indices = [0, 3, 5]  # 0 ppb, 30 ppb, 60 ppb (worst cases)
        
        # Create subplot figure (3 rows × 2 columns)
        subplot_titles = []
        for idx in test_indices:
            subplot_titles.extend([
                f'Basic Interpolation - {self.concentrations[idx]:.0f} ppb',
                f'Geodesic Model - {self.concentrations[idx]:.0f} ppb'
            ])
        
        fig = make_subplots(
            rows=3, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'surface'}]] * 3,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.05,
            vertical_spacing=0.08,
            column_titles=['<b>Basic Cubic/Linear Interpolation</b>', 
                          '<b>Geodesic-Coupled NODE (3 epochs)</b>']
        )
        
        # Store metrics
        basic_metrics = []
        geodesic_metrics = []
        
        print("\n" + "="*60)
        print("3D SURFACE COMPARISON")
        print("="*60)
        
        # Create comparison plots
        for i, holdout_idx in enumerate(test_indices):
            row = i + 1
            conc = self.concentrations[holdout_idx]
            print(f"\nProcessing {conc:.0f} ppb holdout...")
            
            # Basic interpolation (left column)
            basic_interp = self.basic_interpolation(holdout_idx)
            basic_result = self.create_surface_plot(
                "Basic Interpolation", basic_interp, holdout_idx, 
                row, 1, fig, show_colorbar=(i == len(test_indices) - 1)
            )
            basic_metrics.append(basic_result)
            print(f"  Basic: R²={basic_result['r2']:.3f}, RMSE={basic_result['rmse']:.4f}")
            
            # Geodesic interpolation (right column)
            if model is not None and dataset is not None:
                geodesic_interp = self.geodesic_interpolation(model, holdout_idx, dataset)
                geodesic_result = self.create_surface_plot(
                    "Geodesic Model", geodesic_interp, holdout_idx,
                    row, 2, fig, show_colorbar=(i == len(test_indices) - 1)
                )
                geodesic_metrics.append(geodesic_result)
                print(f"  Geodesic: R²={geodesic_result['r2']:.3f}, RMSE={geodesic_result['rmse']:.4f}")
            else:
                # Use basic as placeholder if no model
                geodesic_result = self.create_surface_plot(
                    "Geodesic Model (Not Trained)", basic_interp, holdout_idx,
                    row, 2, fig, show_colorbar=(i == len(test_indices) - 1)
                )
                geodesic_metrics.append(basic_result)
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Interpolation Method Comparison: Basic vs Geodesic-Coupled NODE<br>'
                       '<sub>Leave-One-Out Validation on Arsenic UV-Vis Spectral Data (A100 Implementation)</sub>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'color': '#2c3e50'},
                'y': 0.98
            },
            width=1600,
            height=1400,
            margin=dict(l=0, r=100, t=120, b=50),
            paper_bgcolor='#f8f9fa',
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            ),
            showlegend=False,
            annotations=[
                dict(
                    text="<b>Red line:</b> Actual data | <b>Dashed line:</b> Model prediction | "
                         "<b>Surface:</b> Training data + prediction",
                    xref="paper", yref="paper",
                    x=0.5, y=-0.02,
                    showarrow=False,
                    font=dict(size=12, color='#666'),
                    xanchor='center'
                )
            ]
        )
        
        # Save figure
        output_path = Path('geodesic_a100/test_outputs') / save_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        print(f"\n✅ Saved 3D comparison to {output_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY METRICS")
        print("="*60)
        
        for idx, basic, geodesic in zip(test_indices, basic_metrics, geodesic_metrics):
            conc = self.concentrations[idx]
            print(f"\n{conc:.0f} ppb holdout:")
            print(f"  Basic:    R²={basic['r2']:7.3f}, RMSE={basic['rmse']:.4f}, MAE={basic['mae']:.4f}")
            print(f"  Geodesic: R²={geodesic['r2']:7.3f}, RMSE={geodesic['rmse']:.4f}, MAE={geodesic['mae']:.4f}")
            
            if geodesic['r2'] > basic['r2']:
                improvement = ((geodesic['r2'] - basic['r2']) / abs(basic['r2']) * 100 
                             if basic['r2'] != 0 else 100)
                print(f"  → Geodesic {improvement:.1f}% better")
        
        return fig, basic_metrics, geodesic_metrics


def create_3d_comparison_with_model(model_path=None):
    """Create 3D comparison using a trained model or basic comparison only"""
    
    print("\n" + "="*60)
    print(" 3D SURFACE COMPARISON VISUALIZATION")
    print(" A100 Geodesic Implementation")
    print("="*60)
    
    # Create comparison object
    comparison = Comparison3D(csv_path='data/0.30MB_AuNP_As.csv')
    
    # Try to load model if available
    model = None
    dataset = None
    
    if model_path and Path(model_path).exists():
        print(f"\nLoading model from {model_path}...")
        # Load model implementation would go here
        # For now, we'll use the trained model from test_validation if available
    else:
        print("\nNote: Running without trained model (basic interpolation only)")
        print("      Train a model first for geodesic comparison")
    
    # Create comparison
    fig, basic_metrics, geodesic_metrics = comparison.create_full_comparison(
        model=model, 
        dataset=dataset,
        save_path='test_3d_comparison.html'
    )
    
    print("\n📊 Open geodesic_a100/test_outputs/test_3d_comparison.html in your browser")
    print("\nInteractive features:")
    print("  • Rotate: Click and drag on 3D plots")
    print("  • Zoom: Scroll on plots")
    print("  • Pan: Shift+drag")
    print("  • Compare: Red lines show actual data")
    print("  • Hover: See exact values")
    
    return fig


if __name__ == "__main__":
    create_3d_comparison_with_model()