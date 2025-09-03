#!/usr/bin/env python3
"""
Ultra-Fast Fair Leave-One-Out Comparison
Trains lightweight geodesic models for honest evaluation vs basic interpolation
Target time: ~5 minutes total
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
from pathlib import Path
import time
import json

# Simple lightweight geodesic model for ultra-fast training
class UltraFastGeodesicModel(nn.Module):
    """Ultra-lightweight geodesic model for fast training"""
    
    def __init__(self, device):
        super().__init__()
        self.device = device
        
        # Tiny networks for speed
        self.metric_net = nn.Sequential(
            nn.Linear(2, 8),    # [c, λ] -> 8
            nn.Tanh(),
            nn.Linear(8, 1),    # -> metric g(c,λ)
            nn.Softplus()       # Ensure positive
        ).to(device)
        
        self.spectral_net = nn.Sequential(
            nn.Linear(3, 8),    # [c, v, λ] -> 8  
            nn.Tanh(),
            nn.Linear(8, 1)     # -> dA/dt
        ).to(device)
        
        # Add small constant to avoid division by zero
        self.eps = 1e-6
        
    def forward(self, c_source, c_target, wavelength, A_source):
        """Ultra-simplified forward pass"""
        batch_size = c_source.shape[0]
        
        # Simple linear path in concentration space (no ODE solving for speed)
        n_steps = 3  # Minimal steps
        c_path = torch.linspace(0, 1, n_steps, device=self.device).unsqueeze(0).repeat(batch_size, 1)
        c_path = c_source.unsqueeze(1) + c_path * (c_target - c_source).unsqueeze(1)
        
        # Compute "geodesic-like" prediction
        predictions = []
        for i in range(batch_size):
            c_final = c_target[i]
            wl = wavelength[i]
            A_src = A_source[i]
            
            # Simple spectral flow approximation
            c_input = torch.stack([c_final, wl]).unsqueeze(0)
            spectral_change = self.spectral_net(torch.cat([c_input, torch.zeros(1, 1, device=self.device)], dim=1))
            
            # Predict final absorbance
            A_final = A_src + spectral_change.squeeze()
            predictions.append(A_final)
        
        return {'absorbance': torch.stack(predictions)}

def load_spectral_data(filepath='data/0.30MB_AuNP_As.csv'):
    """Load spectral data"""
    df = pd.read_csv(filepath)
    wavelengths = df['Wavelength'].values
    concentrations = [float(col) for col in df.columns[1:]]
    absorbance_matrix = df.iloc[:, 1:].values
    return wavelengths, concentrations, absorbance_matrix

def basic_interpolation_fair(wavelengths, concentrations, absorbance_matrix, holdout_idx):
    """Fair basic interpolation - excludes holdout from training"""
    train_concs = concentrations[:holdout_idx] + concentrations[holdout_idx+1:]
    train_abs = np.column_stack([absorbance_matrix[:, i] 
                                for i in range(len(concentrations)) 
                                if i != holdout_idx])
    
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

def train_ultra_fast_model(exclude_idx, wavelengths, concentrations, absorbance_matrix, device):
    """Train ultra-fast geodesic model"""
    
    # Get training data (exclude holdout)
    train_indices = [i for i in range(len(concentrations)) if i != exclude_idx]
    train_concs = [concentrations[i] for i in train_indices]
    
    # Create model
    model = UltraFastGeodesicModel(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Ultra-fast training (only 5 epochs, small batches)
    model.train()
    
    for epoch in range(5):  # Ultra-fast training
        total_loss = 0
        n_batches = 0
        
        # Sample wavelengths for speed (every 20th wavelength)
        wl_indices = range(0, len(wavelengths), 20)
        
        for wl_idx in wl_indices:
            wl = wavelengths[wl_idx]
            
            # Create training pairs from available concentrations
            for i, source_idx in enumerate(train_indices):
                for target_idx in train_indices:
                    if source_idx == target_idx:
                        continue
                        
                    # Normalize inputs
                    c_source = torch.tensor([(train_concs[i] - 30) / 30], device=device, dtype=torch.float32)
                    c_target = torch.tensor([(train_concs[train_indices.index(target_idx)] - 30) / 30], device=device, dtype=torch.float32)
                    wavelength_norm = torch.tensor([(wl - 500) / 300], device=device, dtype=torch.float32)
                    
                    A_source = torch.tensor([absorbance_matrix[wl_idx, source_idx]], device=device, dtype=torch.float32)
                    A_target = absorbance_matrix[wl_idx, target_idx]
                    
                    # Forward pass
                    result = model(c_source, c_target, wavelength_norm, A_source)
                    predicted = result['absorbance']
                    target_tensor = torch.tensor([A_target], device=device, dtype=torch.float32)
                    
                    # Compute loss
                    loss = nn.MSELoss()(predicted, target_tensor)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    n_batches += 1
        
        if n_batches > 0:
            avg_loss = total_loss / n_batches
            print(f"    Epoch {epoch+1}/5: Loss={avg_loss:.6f}")
    
    model.eval()
    return model

def geodesic_interpolation_fair(wavelengths, concentrations, absorbance_matrix, holdout_idx, device):
    """Fair geodesic interpolation - trains model excluding holdout"""
    
    print(f"  Training geodesic model excluding {concentrations[holdout_idx]} ppb...")
    model = train_ultra_fast_model(holdout_idx, wavelengths, concentrations, absorbance_matrix, device)
    
    # Predict on holdout concentration
    holdout_conc = concentrations[holdout_idx]
    interpolated_abs = np.zeros(len(wavelengths))
    
    # Find nearest training concentration as source
    train_concs = [concentrations[i] for i in range(len(concentrations)) if i != holdout_idx]
    distances = [abs(tc - holdout_conc) for tc in train_concs]
    nearest_train_conc = train_concs[distances.index(min(distances))]
    nearest_train_idx = concentrations.index(nearest_train_conc)
    
    with torch.no_grad():
        # Process in batches for memory efficiency
        batch_size = 50
        for start in range(0, len(wavelengths), batch_size):
            end = min(start + batch_size, len(wavelengths))
            batch_wls = wavelengths[start:end]
            
            # Normalize inputs
            c_source = torch.full((len(batch_wls),), (nearest_train_conc - 30) / 30, device=device, dtype=torch.float32)
            c_target = torch.full((len(batch_wls),), (holdout_conc - 30) / 30, device=device, dtype=torch.float32)
            wl_norm = torch.tensor([(wl - 500) / 300 for wl in batch_wls], device=device, dtype=torch.float32)
            A_source = torch.tensor(absorbance_matrix[start:end, nearest_train_idx], device=device, dtype=torch.float32)
            
            # Predict
            result = model(c_source, c_target, wl_norm, A_source)
            interpolated_abs[start:end] = result['absorbance'].cpu().numpy()
    
    return interpolated_abs

def compute_metrics(actual, predicted):
    """Compute standard metrics"""
    mse = np.mean((actual - predicted)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    ss_res = np.sum((actual - predicted)**2)
    ss_tot = np.sum((actual - np.mean(actual))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('-inf')
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

def create_comparison_plot(wavelengths, concentrations, absorbance_matrix, basic_results, geodesic_results):
    """Create side-by-side comparison plot"""
    
    test_indices = [1, 3, 5]  # Test on 10, 30, 60 ppb
    
    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}] for _ in range(3)],
        subplot_titles=[
            title for idx in test_indices 
            for title in [f'Basic Interpolation: {concentrations[idx]} ppb', 
                         f'Geodesic Interpolation: {concentrations[idx]} ppb']
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    colors = ['red', 'green', 'blue']
    
    for i, (holdout_idx, basic_result, geodesic_result, color) in enumerate(zip(test_indices, basic_results, geodesic_results, colors)):
        row = i + 1
        conc = concentrations[holdout_idx]
        actual = absorbance_matrix[:, holdout_idx]
        
        # Subsample for clarity
        step = 20
        wl_sub = wavelengths[::step]
        actual_sub = actual[::step]
        basic_sub = basic_result['prediction'][::step]
        geodesic_sub = geodesic_result['prediction'][::step]
        
        # Basic interpolation plot
        fig.add_trace(go.Scatter(
            x=wl_sub, y=actual_sub,
            mode='lines', name=f'Actual {conc} ppb',
            line=dict(color=color, width=3),
            showlegend=(i==0)
        ), row=row, col=1)
        
        fig.add_trace(go.Scatter(
            x=wl_sub, y=basic_sub,
            mode='lines', name=f'Basic Pred',
            line=dict(color=color, width=2, dash='dash'),
            showlegend=(i==0)
        ), row=row, col=1)
        
        # Geodesic interpolation plot
        fig.add_trace(go.Scatter(
            x=wl_sub, y=actual_sub,
            mode='lines', name=f'Actual {conc} ppb',
            line=dict(color=color, width=3),
            showlegend=False
        ), row=row, col=2)
        
        fig.add_trace(go.Scatter(
            x=wl_sub, y=geodesic_sub,
            mode='lines', name=f'Geodesic Pred',
            line=dict(color=color, width=2, dash='dot'),
            showlegend=(i==0)
        ), row=row, col=2)
        
        # Metrics will be printed in console instead of annotations
    
    fig.update_layout(
        title="FAIR Leave-One-Out Comparison: Basic vs Geodesic Interpolation",
        height=800,
        showlegend=True
    )
    
    # Update axis labels
    for i in range(1, 4):
        fig.update_xaxes(title_text="Wavelength (nm)", row=i, col=1)
        fig.update_xaxes(title_text="Wavelength (nm)", row=i, col=2)
        fig.update_yaxes(title_text="Absorbance", row=i, col=1)
        fig.update_yaxes(title_text="Absorbance", row=i, col=2)
    
    return fig

def main():
    """Run ultra-fast fair comparison"""
    
    print("🚀 ULTRA-FAST FAIR LEAVE-ONE-OUT COMPARISON")
    print("="*60)
    print("Target time: ~5 minutes")
    print("Training lightweight geodesic models for honest evaluation")
    print("="*60)
    
    start_time = time.time()
    
    # Setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    wavelengths, concentrations, absorbance_matrix = load_spectral_data()
    print(f"Loaded {len(wavelengths)} wavelengths × {len(concentrations)} concentrations")
    
    # Test on problematic concentrations
    test_indices = [1, 3, 5]  # 10, 30, 60 ppb
    
    basic_results = []
    geodesic_results = []
    
    for i, holdout_idx in enumerate(test_indices):
        holdout_conc = concentrations[holdout_idx]
        actual = absorbance_matrix[:, holdout_idx]
        
        print(f"\n[{i+1}/3] Testing holdout: {holdout_conc} ppb")
        print("-" * 40)
        
        # Basic interpolation (fair)
        print("  Running basic interpolation...")
        basic_pred = basic_interpolation_fair(wavelengths, concentrations, absorbance_matrix, holdout_idx)
        basic_metrics = compute_metrics(actual, basic_pred)
        basic_results.append({'prediction': basic_pred, 'metrics': basic_metrics})
        print(f"    RMSE={basic_metrics['rmse']:.4f}, R²={basic_metrics['r2']:.3f}")
        
        # Geodesic interpolation (fair)  
        geodesic_pred = geodesic_interpolation_fair(wavelengths, concentrations, absorbance_matrix, holdout_idx, device)
        geodesic_metrics = compute_metrics(actual, geodesic_pred)
        geodesic_results.append({'prediction': geodesic_pred, 'metrics': geodesic_metrics})
        print(f"    RMSE={geodesic_metrics['rmse']:.4f}, R²={geodesic_metrics['r2']:.3f}")
    
    # Create visualization
    print("\nCreating comparison visualization...")
    fig = create_comparison_plot(wavelengths, concentrations, absorbance_matrix, basic_results, geodesic_results)
    
    output_file = 'ultra_fast_fair_comparison.html'
    fig.write_html(output_file)
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "="*60)
    print("FAIR COMPARISON RESULTS SUMMARY")
    print("="*60)
    
    test_concentrations = [concentrations[i] for i in test_indices]
    
    print("\nBasic Interpolation (Fair - 5 training points):")
    for conc, result in zip(test_concentrations, basic_results):
        metrics = result['metrics']
        print(f"  {conc:4.0f} ppb: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.3f}")
    
    print("\nGeodesic Interpolation (Fair - 5 training points):")
    for conc, result in zip(test_concentrations, geodesic_results):
        metrics = result['metrics']
        print(f"  {conc:4.0f} ppb: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.3f}")
    
    # Determine winner
    basic_avg_rmse = np.mean([r['metrics']['rmse'] for r in basic_results])
    geodesic_avg_rmse = np.mean([r['metrics']['rmse'] for r in geodesic_results])
    
    print(f"\nAverage RMSE:")
    print(f"  Basic: {basic_avg_rmse:.4f}")
    print(f"  Geodesic: {geodesic_avg_rmse:.4f}")
    
    if geodesic_avg_rmse < basic_avg_rmse:
        improvement = (basic_avg_rmse - geodesic_avg_rmse) / basic_avg_rmse * 100
        print(f"\n✅ Geodesic method wins! {improvement:.1f}% RMSE improvement")
    else:
        degradation = (geodesic_avg_rmse - basic_avg_rmse) / basic_avg_rmse * 100  
        print(f"\n❌ Basic method wins. Geodesic {degradation:.1f}% worse RMSE")
        print("   (Geodesic needs more training or better architecture)")
    
    print(f"\n⏱️  Total time: {total_time:.1f}s")
    print(f"📊 Results saved to: {output_file}")
    print("="*60)

if __name__ == "__main__":
    main()