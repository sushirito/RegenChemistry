#!/usr/bin/env python3
"""
Create the fancy 3D surface comparison plots with beautiful colors
"""

import sys
import os
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'geodesic_m1'))

# Now we can import
from geodesic_m1.visualization.comparison_3d import create_3d_comparison_plot

def main():
    print("🎨 Creating Fancy 3D Surface Comparison Plots...")
    print("=" * 70)
    
    # Use the existing predictions CSV
    predictions_csv = "outputs/predictions/validation_predictions.csv"
    
    # Create the beautiful 3D plots
    print("📊 Generating 3D surface plots with fancy colors...")
    
    fig = create_3d_comparison_plot(
        predictions_csv=predictions_csv,
        data_path="data/0.30MB_AuNP_As.csv",  # Using YOUR REAL DATA
        save_path="outputs/fancy_3d_surface_comparison.html",
        show_plot=True,
        models=None,  # Will use all available models
        checkpoint_dir="checkpoints"
    )
    
    print("✅ Created fancy 3D surface plots!")
    print("💾 Saved to: outputs/fancy_3d_surface_comparison.html")
    
    # Try to open in browser
    import subprocess
    try:
        subprocess.run(['open', 'outputs/fancy_3d_surface_comparison.html'], check=False)
        print("🖼️  Opening in browser...")
    except:
        pass
    
    print("\n" + "=" * 70)
    print("✨ BEAUTIFUL 3D SURFACE PLOTS COMPLETED!")
    print("=" * 70)

if __name__ == "__main__":
    main()