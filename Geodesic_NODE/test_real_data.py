#!/usr/bin/env python3
"""
Test loading REAL data only - no synthetic fallback
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

def load_real_spectral_data():
    """Load the REAL spectral data from CSV"""
    print("📊 Loading REAL spectral data...")
    
    # Load the CSV
    df = pd.read_csv('data/spectral_data.csv')
    print(f"   Data shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # Extract wavelengths and absorbance
    wavelengths = df['Wavelength'].values
    concentrations = np.array([0, 10, 20, 30, 40, 60])
    
    # Get absorbance data for each concentration
    absorbance_matrix = []
    for conc in ['0', '10', '20', '30', '40', '60']:
        absorbance_matrix.append(df[conc].values)
    
    absorbance_matrix = np.array(absorbance_matrix)
    print(f"   Absorbance matrix shape: {absorbance_matrix.shape}")
    print(f"   Wavelength range: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")
    print(f"   Absorbance range: {absorbance_matrix.min():.4f} - {absorbance_matrix.max():.4f}")
    
    # Check for any NaN or infinite values
    print(f"   Contains NaN: {np.isnan(absorbance_matrix).any()}")
    print(f"   Contains Inf: {np.isinf(absorbance_matrix).any()}")
    
    # Visualize the real data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: All spectra
    for i, conc in enumerate(concentrations):
        ax1.plot(wavelengths, absorbance_matrix[i], label=f'{conc} ppb', linewidth=2)
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Absorbance')
    ax1.set_title('REAL Spectral Data - All Concentrations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Heatmap
    im = ax2.imshow(absorbance_matrix, aspect='auto', origin='lower',
                    extent=[wavelengths.min(), wavelengths.max(), 0, 60],
                    cmap='viridis')
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Concentration (ppb)')
    ax2.set_title('REAL Absorbance Heatmap')
    plt.colorbar(im, ax=ax2, label='Absorbance')
    
    plt.tight_layout()
    plt.savefig('outputs/real_data_visualization.png', dpi=150)
    print("   💾 Saved visualization to outputs/real_data_visualization.png")
    plt.show()
    
    return wavelengths, concentrations, absorbance_matrix

if __name__ == "__main__":
    print("🔬 Testing REAL Data Loading")
    print("=" * 60)
    
    wavelengths, concentrations, absorbance_matrix = load_real_spectral_data()
    
    print("\n✅ REAL data loaded successfully!")
    print("=" * 60)
    print("Summary:")
    print(f"  • {len(wavelengths)} wavelengths")
    print(f"  • {len(concentrations)} concentrations: {concentrations}")
    print(f"  • Absorbance matrix: {absorbance_matrix.shape}")
    print(f"  • Data is 100% REAL from your CSV file")
    print("=" * 60)