#!/usr/bin/env python3
"""
Demonstration of Fair Leave-One-Out Validation
Shows the difference between fair and unfair comparisons
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d

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

def compute_metrics(actual, predicted):
    """Compute standard metrics"""
    mse = np.mean((actual - predicted)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    ss_res = np.sum((actual - predicted)**2)
    ss_tot = np.sum((actual - np.mean(actual))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('-inf')
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

def demonstrate_fair_vs_unfair():
    """Demonstrate the difference between fair and unfair comparisons"""
    
    print("🧪 FAIR vs UNFAIR COMPARISON DEMONSTRATION")
    print("="*60)
    
    # Load data
    wavelengths, concentrations, absorbance_matrix = load_spectral_data()
    
    # Test on the problematic 60 ppb concentration (index 5)
    holdout_idx = 5
    holdout_conc = concentrations[holdout_idx]
    actual_spectrum = absorbance_matrix[:, holdout_idx]
    
    print(f"Testing holdout concentration: {holdout_conc} ppb")
    print(f"Available training concentrations: {[c for i, c in enumerate(concentrations) if i != holdout_idx]}")
    
    # FAIR: Basic interpolation using only 5 training points
    fair_prediction = basic_interpolation_fair(wavelengths, concentrations, absorbance_matrix, holdout_idx)
    fair_metrics = compute_metrics(actual_spectrum, fair_prediction)
    
    print("\nFAIR Comparison Results:")
    print(f"  Basic Interpolation (5 training points): RMSE={fair_metrics['rmse']:.4f}, R²={fair_metrics['r2']:.3f}")
    
    # UNFAIR: "Geodesic" that was trained on ALL 6 points including the "holdout"
    # For demonstration, let's simulate what this would look like:
    # It would have much lower error because it "cheated" by seeing the test data
    unfair_error = fair_metrics['rmse'] * 0.6  # Simulate lower error due to cheating
    unfair_r2 = 0.85  # Much better R² because it saw the test data
    
    print(f"  Geodesic Method (CHEATING - saw all 6 points): RMSE={unfair_error:.4f}, R²={unfair_r2:.3f}")
    
    print("\n❌ UNFAIR CONCLUSION: 'Geodesic method is better!'")
    print("✅ REALITY: The geodesic method cheated by training on test data")
    
    # What fair geodesic comparison should look like
    print("\n🎯 What FAIR Geodesic Comparison Should Show:")
    print("  1. Train geodesic model on SAME 5 concentrations as basic method")
    print("  2. Test both methods on unseen 6th concentration") 
    print("  3. Compare performance fairly")
    print("  4. Geodesic should only win if it's genuinely better, not because it cheated")
    
    # Create a simple visualization
    fig = go.Figure()
    
    # Add actual spectrum
    fig.add_trace(go.Scatter(
        x=wavelengths[::10],  # Subsample for clarity
        y=actual_spectrum[::10],
        mode='lines',
        name='Actual (60 ppb)',
        line=dict(color='red', width=3)
    ))
    
    # Add fair prediction
    fig.add_trace(go.Scatter(
        x=wavelengths[::10],
        y=fair_prediction[::10],
        mode='lines',
        name=f'Fair Basic Interpolation (R²={fair_metrics["r2"]:.3f})',
        line=dict(color='blue', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f"Fair Leave-One-Out Validation: {holdout_conc} ppb Holdout",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Absorbance",
        height=400
    )
    
    fig.write_html('fair_comparison_demo.html')
    print(f"\n📊 Visualization saved to: fair_comparison_demo.html")
    
    return fair_metrics

if __name__ == "__main__":
    metrics = demonstrate_fair_vs_unfair()
    
    print("\n" + "="*60)
    print("KEY TAKEAWAY")
    print("="*60)
    print("The previous 'geodesic wins' result was INVALID because:")
    print("1. Basic method: Trained on 5 concentrations, tested on 6th ✅")
    print("2. Geodesic method: Trained on ALL 6 concentrations, 'tested' on 6th ❌")
    print("")
    print("To get valid results, we need to:")
    print("1. Train separate geodesic models for each holdout case")
    print("2. Each model only sees 5 training concentrations")
    print("3. Compare on truly unseen test concentration")
    print("="*60)