#!/usr/bin/env python3
"""
Comparison of Basic Interpolation vs Geodesic Interpolation
Creates side-by-side 3D manifolds showing both methods for leave-one-out validation
"""

import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
from pathlib import Path
import json
import sys

# Add geodesic_mps directory to path for imports
import os
geodesic_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, geodesic_path)

try:
    from models.geodesic_model import ParallelGeodesicModel
    from training.data_generator import SpectralDataGenerator
    from training.train_holdout import train_holdout_model
except ImportError:
    # Try alternative approach for when run as standalone script
    print("Import issue detected. Please run with:")
    print("cd geodesic_mps && python -c \"exec(open('utils/compare_methods.py').read())\"")
    sys.exit(1)


class InterpolationComparison:
    """Compare basic interpolation with learned geodesic interpolation"""
    
    def __init__(self, data_path='data/0.30MB_AuNP_As.csv', 
                 checkpoint_path='checkpoints/best_model.pt',
                 use_fair_comparison=True):
        """Initialize with data and model paths"""
        self.data_path = Path(data_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.use_fair_comparison = use_fair_comparison
        
        # Load data
        self.wavelengths, self.concentrations, self.absorbance_matrix = self.load_spectral_data()
        
        # Load trained model if available (only for unfair comparison)
        self.model = None
        if not use_fair_comparison and self.checkpoint_path.exists():
            self.load_model()
    
    def load_spectral_data(self):
        """Load and reshape spectral data"""
        df = pd.read_csv(self.data_path)
        wavelengths = df['Wavelength'].values
        concentrations = [float(col) for col in df.columns[1:]]
        absorbance_matrix = df.iloc[:, 1:].values
        return wavelengths, concentrations, absorbance_matrix
    
    def load_model(self):
        """Load trained geodesic model"""
        from geodesic_mps.models.geodesic_model import ParallelGeodesicModel
        
        print(f"Loading model from {self.checkpoint_path}...")
        self.model = ParallelGeodesicModel(device=self.device)
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint['model_state']
        
        # Handle torch.compile wrapped state dict
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key[10:]
            else:
                new_key = key
            new_state_dict[new_key] = value
        
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
    
    def basic_interpolation(self, holdout_idx):
        """Perform basic cubic/linear interpolation"""
        train_concs = self.concentrations[:holdout_idx] + self.concentrations[holdout_idx+1:]
        train_abs = np.column_stack([self.absorbance_matrix[:, i] 
                                    for i in range(len(self.concentrations)) 
                                    if i != holdout_idx])
        
        interpolated_abs = np.zeros(len(self.wavelengths))
        holdout_conc = self.concentrations[holdout_idx]
        
        for i, wl in enumerate(self.wavelengths):
            if len(train_concs) > 3:
                interp_func = interp1d(train_concs, train_abs[i, :], kind='cubic',
                                      fill_value='extrapolate', bounds_error=False)
            else:
                interp_func = interp1d(train_concs, train_abs[i, :], kind='linear',
                                      fill_value='extrapolate', bounds_error=False)
            interpolated_abs[i] = interp_func(holdout_conc)
        
        return interpolated_abs
    
    def geodesic_interpolation(self, holdout_idx):
        """Perform fair geodesic interpolation by training a new model"""
        
        # For fair comparison, train a new model excluding this holdout concentration
        if self.use_fair_comparison:
            print(f"  Training geodesic model excluding concentration index {holdout_idx}...")
            model = train_holdout_model(
                exclude_concentration_idx=holdout_idx,
                device=self.device,
                n_epochs=8,  # Quick training for demo
                verbose=False
            )
        else:
            # Use pre-trained model (unfair comparison)
            if self.model is None:
                return None
            model = self.model
        
        holdout_conc = self.concentrations[holdout_idx]
        interpolated_abs = np.zeros(len(self.wavelengths))
        
        # Find nearest neighbors for geodesic interpolation
        train_concs = [self.concentrations[i] for i in range(len(self.concentrations)) 
                      if i != holdout_idx]
        
        # Find closest source concentration
        distances = [abs(tc - holdout_conc) for tc in train_concs]
        nearest_idx = distances.index(min(distances))
        source_conc = train_concs[nearest_idx]
        
        # Get source concentration index in original list
        source_idx = self.concentrations.index(source_conc)
        
        # Normalize concentrations
        source_norm = 2 * (source_conc - 0) / 60 - 1
        target_norm = 2 * (holdout_conc - 0) / 60 - 1
        
        # Process wavelengths in batches for speed
        batch_size = 50
        with torch.no_grad():
            for batch_start in range(0, len(self.wavelengths), batch_size):
                batch_end = min(batch_start + batch_size, len(self.wavelengths))
                batch_wl = self.wavelengths[batch_start:batch_end]
                
                # Normalize wavelengths
                wl_norm = 2 * (batch_wl - 200) / 600 - 1
                
                # Create batch tensors
                n_batch = len(batch_wl)
                c_source = torch.full((n_batch,), source_norm, device=self.device, dtype=torch.float32)
                c_target = torch.full((n_batch,), target_norm, device=self.device, dtype=torch.float32)
                wavelength = torch.tensor(wl_norm, device=self.device, dtype=torch.float32)
                
                # Get source absorbances
                A_source = torch.tensor(
                    self.absorbance_matrix[batch_start:batch_end, source_idx],
                    device=self.device, dtype=torch.float32
                )
                
                # Predict using geodesic model
                result = model(c_source, c_target, wavelength, A_source=A_source)
                interpolated_abs[batch_start:batch_end] = result['absorbance'].cpu().numpy()
        
        return interpolated_abs
    
    def create_comparison_surface(self, method_name, holdout_idx, row, col, fig, 
                                show_colorbar=False):
        """Create a 3D surface plot for a specific interpolation method"""
        
        # Get interpolated values based on method
        if method_name == "Basic Interpolation":
            interpolated_abs = self.basic_interpolation(holdout_idx)
            colorscale = 'Viridis'
            surface_color = 'blue'
        else:  # Geodesic Interpolation
            interpolated_abs = self.geodesic_interpolation(holdout_idx)
            if interpolated_abs is None:
                return None
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
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((self.absorbance_matrix[:, holdout_idx] - interpolated_abs)**2))
        mae = np.mean(np.abs(self.absorbance_matrix[:, holdout_idx] - interpolated_abs))
        max_error = np.max(np.abs(self.absorbance_matrix[:, holdout_idx] - interpolated_abs))
        
        # Calculate R² score
        ss_res = np.sum((self.absorbance_matrix[:, holdout_idx] - interpolated_abs)**2)
        ss_tot = np.sum((self.absorbance_matrix[:, holdout_idx] - 
                        np.mean(self.absorbance_matrix[:, holdout_idx]))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Update subplot
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
        
        return {
            'rmse': rmse,
            'mae': mae,
            'max_error': max_error,
            'r2': r2
        }
    
    def create_full_comparison(self):
        """Create complete comparison visualization"""
        
        # Select key concentrations to compare
        test_indices = [1, 3, 5]  # 10 ppb, 30 ppb, 60 ppb
        
        # Create subplot figure (3 rows × 2 columns)
        subplot_titles = []
        for idx in test_indices:
            subplot_titles.extend([
                f'Basic Interpolation - {self.concentrations[idx]:.0f} ppb',
                f'Geodesic Interpolation - {self.concentrations[idx]:.0f} ppb'
            ])
        
        fig = make_subplots(
            rows=3, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'surface'}]] * 3,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.05,
            vertical_spacing=0.08,
            column_titles=['<b>Basic Cubic/Linear Interpolation</b>', 
                          '<b>Learned Geodesic Interpolation</b>']
        )
        
        # Store metrics for comparison
        basic_metrics = []
        geodesic_metrics = []
        
        # Create comparison plots
        for i, holdout_idx in enumerate(test_indices):
            row = i + 1
            
            # Basic interpolation (left column)
            basic_result = self.create_comparison_surface(
                "Basic Interpolation", holdout_idx, row, 1, fig,
                show_colorbar=(i == len(test_indices) - 1)
            )
            if basic_result:
                basic_metrics.append(basic_result)
            
            # Geodesic interpolation (right column)
            geodesic_result = self.create_comparison_surface(
                "Geodesic Interpolation", holdout_idx, row, 2, fig,
                show_colorbar=(i == len(test_indices) - 1)
            )
            if geodesic_result:
                geodesic_metrics.append(geodesic_result)
        
        # Print metric comparisons to console
        print("\n" + "="*60)
        print("INTERPOLATION COMPARISON METRICS")
        print("="*60)
        for i, (holdout_idx, basic, geodesic) in enumerate(zip(test_indices, basic_metrics, geodesic_metrics)):
            conc = self.concentrations[holdout_idx]
            print(f"\nHoldout Concentration: {conc} ppb")
            print(f"  Basic Interpolation:   RMSE={basic['rmse']:.4f}, MAE={basic['mae']:.4f}, R²={basic['r2']:.3f}")
            print(f"  Geodesic Interpolation: RMSE={geodesic['rmse']:.4f}, MAE={geodesic['mae']:.4f}, R²={geodesic['r2']:.3f}")
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Interpolation Method Comparison: Basic vs Geodesic<br>'
                       '<sub>Leave-One-Out Validation on Arsenic UV-Vis Spectral Data</sub>',
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
                    text="<b>Red line:</b> Actual data | <b>Dashed line:</b> Interpolated prediction",
                    xref="paper", yref="paper",
                    x=0.5, y=-0.02,
                    showarrow=False,
                    font=dict(size=12, color='#666'),
                    xanchor='center'
                )
            ]
        )
        
        return fig, basic_metrics, geodesic_metrics
    
    def create_metric_summary_plot(self, basic_metrics, geodesic_metrics):
        """Create a summary plot comparing metrics"""
        
        concentrations = [self.concentrations[i] for i in [1, 3, 5]]
        
        fig = go.Figure()
        
        # RMSE comparison
        fig.add_trace(go.Bar(
            name='Basic Interpolation',
            x=concentrations,
            y=[m['rmse'] for m in basic_metrics],
            marker_color='blue',
            opacity=0.7
        ))
        
        fig.add_trace(go.Bar(
            name='Geodesic Interpolation',
            x=concentrations,
            y=[m['rmse'] for m in geodesic_metrics],
            marker_color='purple',
            opacity=0.7
        ))
        
        fig.update_layout(
            title='RMSE Comparison: Basic vs Geodesic Interpolation',
            xaxis_title='Held-out Concentration (ppb)',
            yaxis_title='RMSE',
            barmode='group',
            template='plotly_white',
            width=800,
            height=400,
            showlegend=True
        )
        
        return fig


def main():
    """Main function to create comparison visualizations"""
    
    print("="*60)
    print("INTERPOLATION METHOD COMPARISON")
    print("Basic vs Geodesic Interpolation")
    print("="*60)
    
    # Create comparison object with fair leave-one-out validation
    comparison = InterpolationComparison(use_fair_comparison=True)
    
    print("Using FAIR leave-one-out validation:")
    print("  - Basic method: Only uses 5 training concentrations")
    print("  - Geodesic method: Trains fresh model on same 5 concentrations")
    print("  - No cheating: Neither method sees the holdout concentration")
    
    print("\nGenerating comparison visualizations...")
    
    # Create main comparison
    fig_3d, basic_metrics, geodesic_metrics = comparison.create_full_comparison()
    
    # Save 3D comparison
    output_3d = 'logs/interpolation_comparison_3d.html'
    fig_3d.write_html(output_3d)
    print(f"✅ Saved 3D comparison to {output_3d}")
    
    # Create metrics summary
    fig_metrics = comparison.create_metric_summary_plot(basic_metrics, geodesic_metrics)
    output_metrics = 'logs/interpolation_metrics.html'
    fig_metrics.write_html(output_metrics)
    print(f"✅ Saved metrics comparison to {output_metrics}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("VALIDATION RESULTS SUMMARY")
    print("="*60)
    
    test_concentrations = [comparison.concentrations[i] for i in [1, 3, 5]]
    
    print("\nBasic Interpolation:")
    for conc, metrics in zip(test_concentrations, basic_metrics):
        print(f"  {conc:5.0f} ppb: RMSE={metrics['rmse']:.4f}, "
              f"MAE={metrics['mae']:.4f}, R²={metrics['r2']:.3f}")
    avg_rmse_basic = np.mean([m['rmse'] for m in basic_metrics])
    avg_r2_basic = np.mean([m['r2'] for m in basic_metrics])
    print(f"  Average: RMSE={avg_rmse_basic:.4f}, R²={avg_r2_basic:.3f}")
    
    print("\nGeodesic Interpolation:")
    for conc, metrics in zip(test_concentrations, geodesic_metrics):
        print(f"  {conc:5.0f} ppb: RMSE={metrics['rmse']:.4f}, "
              f"MAE={metrics['mae']:.4f}, R²={metrics['r2']:.3f}")
    avg_rmse_geodesic = np.mean([m['rmse'] for m in geodesic_metrics])
    avg_r2_geodesic = np.mean([m['r2'] for m in geodesic_metrics])
    print(f"  Average: RMSE={avg_rmse_geodesic:.4f}, R²={avg_r2_geodesic:.3f}")
    
    print("\n" + "="*60)
    print("IMPROVEMENT SUMMARY")
    print("="*60)
    
    rmse_improvement = (avg_rmse_basic - avg_rmse_geodesic) / avg_rmse_basic * 100
    r2_improvement = avg_r2_geodesic - avg_r2_basic
    
    print(f"RMSE Improvement: {rmse_improvement:.1f}% reduction")
    print(f"R² Improvement: {r2_improvement:+.3f}")
    
    if rmse_improvement > 0:
        print("\n✅ Geodesic interpolation shows improvement over basic method!")
    else:
        print("\n⚠️  Basic interpolation currently performs better.")
        print("   More training may be needed for the geodesic model.")
    
    print("\n" + "="*60)
    print("\n📊 Open the following files in your browser:")
    print(f"   • {output_3d} - Interactive 3D comparison")
    print(f"   • {output_metrics} - Metrics bar chart")
    print("\nInteractive features:")
    print("  • Rotate: Click and drag on 3D plots")
    print("  • Zoom: Scroll on plots")
    print("  • Compare: Red lines show actual data")
    print("  • Metrics: Hover for exact values")


if __name__ == "__main__":
    main()