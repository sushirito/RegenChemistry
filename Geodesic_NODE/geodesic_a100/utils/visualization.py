"""
Visualization utilities for A100 Geodesic NODE testing
Adapted from geodesic_mps but for coupled ODE system
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd


class Visualizer:
    """Create visualizations for geodesic model performance"""
    
    def __init__(self, output_dir: str = "geodesic_a100/test_outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_geodesic_paths(self, model, n_samples: int = 5,
                           save_name: str = "test_geodesic_paths.html"):
        """
        Visualize geodesic paths showing coupled dynamics [c, v, A]
        
        Creates 2x2 subplot:
        - Concentration trajectories c(t)
        - Velocity profiles v(t)
        - Absorbance evolution A(t)
        - Metric values g(c,λ) along path
        """
        device = next(model.parameters()).device
        
        # Test cases: various concentration transitions
        c_source = torch.tensor([-0.8], device=device)  # Normalized
        c_targets = torch.linspace(-0.5, 0.5, n_samples, device=device)
        wavelength = torch.tensor([0.0], device=device)  # Center wavelength
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Concentration Paths c(t)",
                "Velocity Profiles v(t)",
                "Absorbance Evolution A(t)",
                "Metric g(c,λ) Along Path"
            )
        )
        
        colors = plt.cm.viridis(np.linspace(0, 1, n_samples))
        
        for i, c_target in enumerate(c_targets):
            with torch.no_grad():
                result = model(
                    c_source.unsqueeze(0),
                    c_target.unsqueeze(0),
                    wavelength.unsqueeze(0)
                )
            
            if 'trajectories' in result and result['trajectories'] is not None:
                # Trajectories shape: [n_points, batch_size=1, 3]
                traj = result['trajectories'].squeeze(1).cpu().numpy()  # [n_points, 3]
                t = np.linspace(0, 1, len(traj))
                
                color_str = f'rgb({colors[i][0]*255:.0f},{colors[i][1]*255:.0f},{colors[i][2]*255:.0f})'
                
                # Concentration path
                fig.add_trace(
                    go.Scatter(
                        x=t, y=traj[:, 0],
                        mode='lines',
                        name=f'Target={c_target.item():.2f}',
                        line=dict(color=color_str, width=2)
                    ),
                    row=1, col=1
                )
                
                # Velocity profile
                fig.add_trace(
                    go.Scatter(
                        x=t, y=traj[:, 1],
                        mode='lines',
                        showlegend=False,
                        line=dict(color=color_str, width=2)
                    ),
                    row=1, col=2
                )
                
                # Absorbance evolution (coupled ODE output)
                fig.add_trace(
                    go.Scatter(
                        x=t, y=traj[:, 2],
                        mode='lines',
                        showlegend=False,
                        line=dict(color=color_str, width=2)
                    ),
                    row=2, col=1
                )
                
                # Compute metric along path
                c_path = torch.tensor(traj[:, 0], device=device, dtype=torch.float32)
                wl_path = wavelength.repeat(len(c_path))
                inputs = torch.stack([c_path, wl_path], dim=1)
                
                with torch.no_grad():
                    g_values = model.metric_network(inputs).squeeze().cpu().numpy()
                
                fig.add_trace(
                    go.Scatter(
                        x=t, y=g_values,
                        mode='lines',
                        showlegend=False,
                        line=dict(color=color_str, width=2)
                    ),
                    row=2, col=2
                )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time t", row=1, col=1)
        fig.update_xaxes(title_text="Time t", row=1, col=2)
        fig.update_xaxes(title_text="Time t", row=2, col=1)
        fig.update_xaxes(title_text="Time t", row=2, col=2)
        
        fig.update_yaxes(title_text="Concentration (norm)", row=1, col=1)
        fig.update_yaxes(title_text="Velocity dc/dt", row=1, col=2)
        fig.update_yaxes(title_text="Absorbance", row=2, col=1)
        fig.update_yaxes(title_text="Metric g(c,λ)", row=2, col=2)
        
        fig.update_layout(
            title="Geodesic Paths with Coupled Dynamics [c, v, A]",
            height=800,
            showlegend=True
        )
        
        save_path = self.output_dir / save_name
        fig.write_html(save_path)
        print(f"  Saved geodesic paths to {save_path}")
        return fig
    
    def plot_comparison(self, basic_results: Dict, geodesic_results: Dict,
                       wavelengths: np.ndarray, save_name: str = "test_comparison.html"):
        """
        Compare basic interpolation vs geodesic model performance
        
        Args:
            basic_results: Dict with 'predictions', 'actual', 'metrics'
            geodesic_results: Dict with 'predictions', 'actual', 'metrics'
            wavelengths: Wavelength array for coloring
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Basic Interpolation: Predicted vs Actual",
                "Geodesic Model: Predicted vs Actual",
                "Basic Interpolation: Residuals",
                "Geodesic Model: Residuals"
            )
        )
        
        # Basic interpolation scatter
        fig.add_trace(
            go.Scatter(
                x=basic_results['actual'],
                y=basic_results['predictions'],
                mode='markers',
                marker=dict(
                    size=3,
                    color=wavelengths,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="λ (nm)", x=0.45)
                ),
                name='Basic'
            ),
            row=1, col=1
        )
        
        # Perfect prediction line for basic
        min_val = min(basic_results['actual'].min(), basic_results['predictions'].min())
        max_val = max(basic_results['actual'].max(), basic_results['predictions'].max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Geodesic model scatter
        fig.add_trace(
            go.Scatter(
                x=geodesic_results['actual'],
                y=geodesic_results['predictions'],
                mode='markers',
                marker=dict(
                    size=3,
                    color=wavelengths,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="λ (nm)", x=1.02)
                ),
                name='Geodesic'
            ),
            row=1, col=2
        )
        
        # Perfect prediction line for geodesic
        min_val = min(geodesic_results['actual'].min(), geodesic_results['predictions'].min())
        max_val = max(geodesic_results['actual'].max(), geodesic_results['predictions'].max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Residuals for basic
        basic_residuals = basic_results['predictions'] - basic_results['actual']
        fig.add_trace(
            go.Scatter(
                x=basic_results['actual'],
                y=basic_residuals,
                mode='markers',
                marker=dict(
                    size=3,
                    color=wavelengths,
                    colorscale='Viridis',
                    showscale=False
                ),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Zero line for basic residuals
        fig.add_trace(
            go.Scatter(
                x=[basic_results['actual'].min(), basic_results['actual'].max()],
                y=[0, 0],
                mode='lines',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Residuals for geodesic
        geodesic_residuals = geodesic_results['predictions'] - geodesic_results['actual']
        fig.add_trace(
            go.Scatter(
                x=geodesic_results['actual'],
                y=geodesic_residuals,
                mode='markers',
                marker=dict(
                    size=3,
                    color=wavelengths,
                    colorscale='Viridis',
                    showscale=False
                ),
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Zero line for geodesic residuals
        fig.add_trace(
            go.Scatter(
                x=[geodesic_results['actual'].min(), geodesic_results['actual'].max()],
                y=[0, 0],
                mode='lines',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update axes
        fig.update_xaxes(title_text="Actual Absorbance", row=1, col=1)
        fig.update_xaxes(title_text="Actual Absorbance", row=1, col=2)
        fig.update_xaxes(title_text="Actual Absorbance", row=2, col=1)
        fig.update_xaxes(title_text="Actual Absorbance", row=2, col=2)
        
        fig.update_yaxes(title_text="Predicted", row=1, col=1)
        fig.update_yaxes(title_text="Predicted", row=1, col=2)
        fig.update_yaxes(title_text="Residual", row=2, col=1)
        fig.update_yaxes(title_text="Residual", row=2, col=2)
        
        # Add metrics to title
        basic_r2 = basic_results['metrics']['r2']
        basic_mse = basic_results['metrics']['mse']
        geodesic_r2 = geodesic_results['metrics']['r2']
        geodesic_mse = geodesic_results['metrics']['mse']
        
        title = (f"Basic Interpolation (R²={basic_r2:.3f}, MSE={basic_mse:.4f}) vs "
                f"Geodesic Model (R²={geodesic_r2:.3f}, MSE={geodesic_mse:.4f})")
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=False
        )
        
        save_path = self.output_dir / save_name
        fig.write_html(save_path)
        print(f"  Saved comparison to {save_path}")
        return fig
    
    def plot_metric_landscape(self, model, n_points: int = 50,
                            save_name: str = "test_metric_landscape.html"):
        """Visualize the learned metric as a 3D surface"""
        device = next(model.parameters()).device
        
        # Create grid
        c_range = torch.linspace(-1, 1, n_points, device=device)
        wl_range = torch.linspace(-1, 1, n_points, device=device)
        
        # Compute metric on grid
        g_grid = np.zeros((n_points, n_points))
        
        with torch.no_grad():
            for i, c in enumerate(c_range):
                for j, wl in enumerate(wl_range):
                    inputs = torch.tensor([[c.item(), wl.item()]], 
                                         device=device, dtype=torch.float32)
                    g_grid[i, j] = model.metric_network(inputs).item()
        
        # Create 3D surface plot
        fig = go.Figure(data=[
            go.Surface(
                x=c_range.cpu().numpy(),
                y=wl_range.cpu().numpy(),
                z=g_grid.T,
                colorscale='Viridis',
                name='Metric g(c,λ)'
            )
        ])
        
        fig.update_layout(
            title="Learned Riemannian Metric g(c,λ)",
            scene=dict(
                xaxis_title="Concentration (normalized)",
                yaxis_title="Wavelength (normalized)",
                zaxis_title="Metric Value g(c,λ)",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=700
        )
        
        save_path = self.output_dir / save_name
        fig.write_html(save_path)
        print(f"  Saved metric landscape to {save_path}")
        return fig
    
    def plot_training_progress(self, train_losses: list, val_losses: list = None,
                             save_name: str = "test_training_progress.html"):
        """Plot training loss curves"""
        epochs = list(range(1, len(train_losses) + 1))
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=train_losses,
                mode='lines+markers',
                name='Training Loss',
                line=dict(color='blue', width=2)
            )
        )
        
        if val_losses:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=val_losses,
                    mode='lines+markers',
                    name='Validation Loss',
                    line=dict(color='red', width=2)
                )
            )
        
        fig.update_layout(
            title=f"Training Progress ({len(epochs)} epochs)",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            hovermode='x unified',
            height=400
        )
        
        save_path = self.output_dir / save_name
        fig.write_html(save_path)
        print(f"  Saved training progress to {save_path}")
        return fig