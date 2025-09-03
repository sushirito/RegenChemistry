#!/usr/bin/env python3
"""
Visualization utilities for Geodesic Spectral Model
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from typing import Dict, Optional, List


class TrainingVisualizer:
    """Create visualizations for training progress and results"""
    
    def __init__(self, log_dir: str = "logs", checkpoint_dir: str = "checkpoints"):
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training loss curves from logs"""
        
        # Load training results
        results_path = self.log_dir / "training_results.json"
        if not results_path.exists():
            print(f"No training results found at {results_path}")
            return
            
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Create figure
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=("Training Progress",)
        )
        
        # Plot if we have data
        if 'train_losses' in results and results['train_losses']:
            epochs = list(range(1, len(results['train_losses']) + 1))
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=results['train_losses'],
                    mode='lines+markers',
                    name='Training Loss',
                    line=dict(color='blue', width=2)
                )
            )
        
        if 'val_losses' in results and results['val_losses']:
            epochs = list(range(1, len(results['val_losses']) + 1))
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=results['val_losses'],
                    mode='lines+markers',
                    name='Validation Loss',
                    line=dict(color='red', width=2)
                )
            )
        
        fig.update_layout(
            title="Training Progress",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            hovermode='x unified',
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.write_html(self.log_dir / "training_curves.html")
            
        return fig
    
    def plot_geodesic_paths(self, model, n_samples: int = 5, 
                           save_path: Optional[str] = None):
        """Visualize geodesic paths in concentration space"""
        
        device = next(model.parameters()).device
        
        # Create test data
        c_source = torch.tensor([-0.8], device=device)
        c_targets = torch.linspace(-0.5, 0.5, n_samples, device=device)
        wavelength = torch.tensor([0.0], device=device)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Geodesic Paths in Concentration Space",
                "Velocity Profiles",
                "Absorbance Evolution",
                "Metric Values Along Path"
            )
        )
        
        colors = plt.cm.viridis(np.linspace(0, 1, n_samples))
        
        for i, c_target in enumerate(c_targets):
            # Get trajectory
            with torch.no_grad():
                result = model(
                    c_source,
                    c_target.unsqueeze(0),
                    wavelength,
                    return_trajectories=True
                )
            
            if 'trajectories' in result and result['trajectories'] is not None:
                traj = result['trajectories'].squeeze().cpu().numpy()
                t = np.linspace(0, 1, len(traj))
                
                # Concentration path
                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=traj[:, 0],
                        mode='lines',
                        name=f'Target={c_target.item():.2f}',
                        line=dict(color=f'rgb({colors[i][0]*255:.0f},{colors[i][1]*255:.0f},{colors[i][2]*255:.0f})')
                    ),
                    row=1, col=1
                )
                
                # Velocity profile
                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=traj[:, 1],
                        mode='lines',
                        showlegend=False,
                        line=dict(color=f'rgb({colors[i][0]*255:.0f},{colors[i][1]*255:.0f},{colors[i][2]*255:.0f})')
                    ),
                    row=1, col=2
                )
                
                # Absorbance evolution
                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=traj[:, 2],
                        mode='lines',
                        showlegend=False,
                        line=dict(color=f'rgb({colors[i][0]*255:.0f},{colors[i][1]*255:.0f},{colors[i][2]*255:.0f})')
                    ),
                    row=2, col=1
                )
                
                # Compute metric along path
                c_path = torch.tensor(traj[:, 0], device=device, dtype=torch.float32)
                wl_path = wavelength.repeat(len(c_path))
                with torch.no_grad():
                    g_values = model.metric_network(c_path, wl_path).cpu().numpy()
                
                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=g_values,
                        mode='lines',
                        showlegend=False,
                        line=dict(color=f'rgb({colors[i][0]*255:.0f},{colors[i][1]*255:.0f},{colors[i][2]*255:.0f})')
                    ),
                    row=2, col=2
                )
        
        fig.update_xaxes(title_text="Time t", row=1, col=1)
        fig.update_xaxes(title_text="Time t", row=1, col=2)
        fig.update_xaxes(title_text="Time t", row=2, col=1)
        fig.update_xaxes(title_text="Time t", row=2, col=2)
        
        fig.update_yaxes(title_text="Concentration", row=1, col=1)
        fig.update_yaxes(title_text="Velocity", row=1, col=2)
        fig.update_yaxes(title_text="Absorbance", row=2, col=1)
        fig.update_yaxes(title_text="Metric g(c,λ)", row=2, col=2)
        
        fig.update_layout(height=800, showlegend=True)
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.write_html(self.log_dir / "geodesic_paths.html")
            
        return fig
    
    def plot_metric_landscape(self, model, n_points: int = 50,
                            save_path: Optional[str] = None):
        """Visualize the learned metric as a 2D landscape"""
        
        device = next(model.parameters()).device
        
        # Create grid
        c_range = torch.linspace(-1, 1, n_points, device=device)
        wl_range = torch.linspace(-1, 1, n_points, device=device)
        
        # Compute metric on grid
        with torch.no_grad():
            g_grid = model.metric_network.forward_parallel(c_range, wl_range)
            g_grid = g_grid.cpu().numpy()
        
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
            title="Learned Riemannian Metric Landscape",
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
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.write_html(self.log_dir / "metric_landscape.html")
            
        return fig
    
    def plot_predictions_vs_actual(self, model, data_generator,
                                  n_samples: int = 100,
                                  save_path: Optional[str] = None):
        """Plot model predictions vs actual values"""
        
        device = next(model.parameters()).device
        
        # Get validation data
        val_data = data_generator.get_validation_set(n_samples=n_samples)
        
        # Get predictions
        model.eval()
        with torch.no_grad():
            predictions = model.forward_batch(val_data)
            predictions = predictions.cpu().numpy()
        
        targets = val_data['target_absorbance'].cpu().numpy()
        
        # Create scatter plot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Predictions vs Actual", "Residuals")
        )
        
        # Scatter plot
        fig.add_trace(
            go.Scatter(
                x=targets,
                y=predictions,
                mode='markers',
                marker=dict(
                    size=5,
                    color=val_data['wavelength_idx'].cpu().numpy(),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Wavelength Index")
                ),
                name='Predictions'
            ),
            row=1, col=1
        )
        
        # Perfect prediction line
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Perfect Prediction'
            ),
            row=1, col=1
        )
        
        # Residuals
        residuals = predictions - targets
        fig.add_trace(
            go.Scatter(
                x=targets,
                y=residuals,
                mode='markers',
                marker=dict(
                    size=5,
                    color=val_data['wavelength_idx'].cpu().numpy(),
                    colorscale='Viridis',
                    showscale=False
                ),
                name='Residuals'
            ),
            row=1, col=2
        )
        
        # Zero line for residuals
        fig.add_trace(
            go.Scatter(
                x=[targets.min(), targets.max()],
                y=[0, 0],
                mode='lines',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Actual Absorbance", row=1, col=1)
        fig.update_yaxes(title_text="Predicted Absorbance", row=1, col=1)
        fig.update_xaxes(title_text="Actual Absorbance", row=1, col=2)
        fig.update_yaxes(title_text="Residual", row=1, col=2)
        
        # Calculate metrics
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        r2 = 1 - (np.sum((targets - predictions) ** 2) / np.sum((targets - targets.mean()) ** 2))
        
        fig.update_layout(
            title=f"Model Performance (MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.4f})",
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.write_html(self.log_dir / "predictions.html")
            
        return fig


def visualize_training_results(model_path: Optional[str] = None):
    """
    Main function to generate all visualizations
    
    Args:
        model_path: Path to saved model checkpoint
    """
    print("Generating visualizations...")
    
    # Create visualizer
    viz = TrainingVisualizer()
    
    # Plot training curves
    print("1. Creating training curves...")
    viz.plot_training_curves()
    print(f"   Saved to {viz.log_dir}/training_curves.html")
    
    # Load model if checkpoint exists
    checkpoint_path = Path(model_path) if model_path else Path("checkpoints/best_model.pt")
    
    if checkpoint_path.exists():
        print(f"\n2. Loading model from {checkpoint_path}...")
        
        # Load model
        from geodesic_mps.models.geodesic_model import ParallelGeodesicModel
        from geodesic_mps.training.data_generator import SpectralDataGenerator
        
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Create model
        model = ParallelGeodesicModel(device=device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle torch.compile wrapped state dict
        state_dict = checkpoint['model_state']
        # Remove '_orig_mod.' prefix if present (from torch.compile)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key[10:]  # Remove '_orig_mod.' prefix
            else:
                new_key = key
            new_state_dict[new_key] = value
        
        model.load_state_dict(new_state_dict)
        model.eval()
        
        # Create data generator
        data_gen = SpectralDataGenerator(device)
        
        print("3. Creating geodesic path visualization...")
        viz.plot_geodesic_paths(model)
        print(f"   Saved to {viz.log_dir}/geodesic_paths.html")
        
        print("4. Creating metric landscape...")
        viz.plot_metric_landscape(model)
        print(f"   Saved to {viz.log_dir}/metric_landscape.html")
        
        print("5. Creating predictions vs actual plot...")
        viz.plot_predictions_vs_actual(model, data_gen)
        print(f"   Saved to {viz.log_dir}/predictions.html")
        
    else:
        print(f"\nNo checkpoint found at {checkpoint_path}")
        print("Run training first to generate model checkpoint")
    
    print("\n✅ Visualization complete!")
    print(f"Open the HTML files in {viz.log_dir}/ to view the plots")


if __name__ == "__main__":
    visualize_training_results()