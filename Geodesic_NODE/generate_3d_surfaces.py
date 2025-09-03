#!/usr/bin/env python3
"""
Generate beautiful 3D surface comparison plots
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add geodesic_m1 to path
sys.path.append(str(Path(__file__).parent / 'geodesic_m1'))

from visualization.comparison_3d import create_3d_comparison_plot
from data.generator import create_full_dataset
from models.geodesic_model import GeodesicNODE

def main():
    print("🎨 Creating Beautiful 3D Surface Comparison Plots...")
    print("=" * 70)
    
    # Setup
    device = torch.device('cpu')  # Use CPU for compatibility
    
    # Load dataset
    wavelengths, absorbance_data = create_full_dataset(
        device=device,
        use_synthetic=False,  # Try real data first
        data_path="data/spectral_data.csv"
    )
    
    # Known concentrations
    concentrations = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 60.0])
    
    # Create model
    model = GeodesicNODE(
        metric_hidden_dims=[128, 256],
        flow_hidden_dims=[64, 128],
        n_trajectory_points=50,
        shooting_max_iter=10,
        shooting_tolerance=1e-4,
        shooting_learning_rate=0.5,
        christoffel_grid_size=(500, 601),
        device=device,
        concentrations=concentrations,
        wavelengths=wavelengths.cpu().numpy(),
        absorbance_matrix=absorbance_data.cpu().numpy()
    )
    
    # Load trained weights
    try:
        checkpoint = torch.load("checkpoints/best_model_0.pt", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Loaded trained model")
    except:
        print("⚠️  Using untrained model")
    
    # Precompute grid
    model.precompute_christoffel_grid()
    model.eval()
    
    # Test different concentration transitions
    test_transitions = [
        (20, "20 ppb"),
        (30, "30 ppb"),
        (40, "40 ppb"),
        (60, "60 ppb")
    ]
    
    for holdout_conc, label in test_transitions:
        print(f"\n📊 Creating 3D surface plot for {label}...")
        
        # Create the beautiful 3D comparison plot
        fig = create_3d_comparison_plot(
            model=model,
            wavelengths=wavelengths.cpu().numpy(),
            concentrations=concentrations,
            absorbance_data=absorbance_data.cpu().numpy(),
            holdout_conc=holdout_conc,
            n_wavelength_samples=30,  # Sampling for surface
            n_conc_samples=20,        # Grid density
            save_path=f"outputs/3d_surface_comparison_{holdout_conc}ppb.html"
        )
        
        print(f"   ✅ Saved to outputs/3d_surface_comparison_{holdout_conc}ppb.html")
        
        # Show the plot
        try:
            fig.show()
        except:
            pass
    
    # Open the first one in browser
    import subprocess
    try:
        subprocess.run(['open', 'outputs/3d_surface_comparison_20ppb.html'], check=False)
        print("\n🖼️  Opening 3D surface visualization in browser...")
    except:
        pass
    
    print("\n" + "=" * 70)
    print("✅ BEAUTIFUL 3D SURFACE PLOTS COMPLETED!")
    print("=" * 70)
    print("📊 Generated visualizations:")
    for conc, _ in test_transitions:
        print(f"   • outputs/3d_surface_comparison_{conc}ppb.html")
    print("\n💡 The plots show geodesic vs basic interpolation surfaces")
    print("   with the actual spectral data points overlaid!")
    print("=" * 70)

if __name__ == "__main__":
    main()