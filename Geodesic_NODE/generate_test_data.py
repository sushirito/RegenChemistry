"""
Generate synthetic test data for debugging
Creates a realistic spectral dataset for testing
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_spectral_data(save_path: str = "data/test_spectral_data.csv"):
    """Generate synthetic arsenic spectral data for debugging"""
    
    # Create output directory
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Parameters
    wavelengths = np.linspace(200, 800, 301)  # 301 points for manageable size
    concentrations = [0, 10, 20, 30, 40, 60]  # ppb
    
    # Generate realistic spectral response
    absorbance_data = np.zeros((len(wavelengths), len(concentrations)))
    
    for i, wavelength in enumerate(wavelengths):
        for j, conc in enumerate(concentrations):
            
            # Main peak around 450nm with concentration dependence
            peak_response = 0.8 * np.exp(-(wavelength - 450)**2 / (2 * 80**2))
            
            # Secondary peak around 350nm  
            secondary_peak = 0.3 * np.exp(-(wavelength - 350)**2 / (2 * 60**2))
            
            # Background absorption
            background = 0.1 * np.exp(-(wavelength - 600)**2 / (2 * 200**2))
            
            # Concentration scaling with nonlinearity
            conc_factor = conc / 60.0  # Normalize to [0, 1]
            nonlinear_scaling = conc_factor + 0.3 * conc_factor**2  # Add nonlinearity
            
            # Wavelength-dependent sensitivity (creates the challenging geometry)
            sensitivity = 1.0 + 0.5 * np.sin((wavelength - 400) / 100)
            
            # Combine all effects
            absorbance = nonlinear_scaling * (peak_response + secondary_peak + background) * sensitivity
            
            # Add some noise for realism
            noise = 0.02 * np.random.normal(0, 1)
            absorbance += noise
            
            # Ensure non-negative
            absorbance = max(0, absorbance)
            
            absorbance_data[i, j] = absorbance
    
    # Create DataFrame
    column_names = ['Wavelength'] + [str(c) for c in concentrations]
    df = pd.DataFrame(absorbance_data, columns=column_names[1:])
    df.insert(0, 'Wavelength', wavelengths)
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    
    print(f"✅ Generated synthetic spectral data: {save_path}")
    print(f"   📏 Wavelengths: {len(wavelengths)} points ({wavelengths[0]}-{wavelengths[-1]} nm)")
    print(f"   🎯 Concentrations: {concentrations} ppb")
    print(f"   📊 Data shape: {absorbance_data.shape}")
    print(f"   📈 Absorbance range: [{absorbance_data.min():.3f}, {absorbance_data.max():.3f}]")
    
    # Show peak wavelengths for each concentration
    print(f"\\n📍 Peak wavelengths by concentration:")
    for j, conc in enumerate(concentrations):
        peak_idx = np.argmax(absorbance_data[:, j])
        peak_wavelength = wavelengths[peak_idx]
        peak_absorbance = absorbance_data[peak_idx, j]
        print(f"   {conc:2.0f} ppb: {peak_wavelength:.1f} nm (A={peak_absorbance:.3f})")
    
    return df


def verify_data_quality(df: pd.DataFrame):
    """Verify the generated data has expected properties"""
    print(f"\\n🔍 DATA QUALITY CHECK:")
    
    wavelengths = df['Wavelength'].values
    concentrations = [float(col) for col in df.columns[1:]]
    absorbance_matrix = df.iloc[:, 1:].values
    
    # Check monotonicity at peak wavelength
    peak_wl_idx = np.argmax(absorbance_matrix[:, -1])  # Peak for highest concentration
    peak_absorbances = absorbance_matrix[peak_wl_idx, :]
    
    print(f"   📍 Peak wavelength: {wavelengths[peak_wl_idx]:.1f} nm")
    print(f"   📈 Peak absorbances: {peak_absorbances}")
    
    # Check if generally increasing with concentration
    is_mostly_increasing = np.sum(np.diff(peak_absorbances) > 0) >= len(peak_absorbances) - 2
    print(f"   ✅ Mostly increasing with concentration: {is_mostly_increasing}")
    
    # Check for non-monotonic regions (the challenging part)
    non_monotonic_count = 0
    for i in range(len(wavelengths)):
        spectrum = absorbance_matrix[i, :]
        diffs = np.diff(spectrum)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        if sign_changes > 0:
            non_monotonic_count += 1
    
    print(f"   🌊 Non-monotonic wavelengths: {non_monotonic_count}/{len(wavelengths)} ({non_monotonic_count/len(wavelengths)*100:.1f}%)")
    
    if non_monotonic_count > 0:
        print(f"   ✅ Good! Non-monotonic behavior will test geodesic capability")
    else:
        print(f"   ⚠️ Warning: All wavelengths monotonic - may be too easy")


if __name__ == "__main__":
    # Generate test data
    df = generate_synthetic_spectral_data()
    
    # Verify quality
    verify_data_quality(df)
    
    print(f"\\n✅ Test data ready for debugging!")