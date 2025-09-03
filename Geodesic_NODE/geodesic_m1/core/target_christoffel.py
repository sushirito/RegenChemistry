"""
Target Christoffel symbol computation from known spectral data
Implements inverse geodesic formulation: force spectral curves to be geodesics
"""

import torch
import numpy as np
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from typing import Dict, Tuple, Optional
import time


class TargetChristoffelComputer:
    """
    Computes target Christoffel symbols by treating known spectral curves as geodesics.
    Uses the principle: if A(c) curves are geodesics, then we can compute the required
    Riemannian metric that makes this true.
    """
    
    def __init__(self,
                 concentrations: np.ndarray,
                 wavelengths: np.ndarray, 
                 absorbance_data: np.ndarray,
                 device: torch.device = torch.device('mps'),
                 smoothing_factor: float = 1e-3):
        """
        Initialize target Christoffel computer
        
        Args:
            concentrations: Known concentration values [n_concs]
            wavelengths: Wavelength values [n_wavelengths] 
            absorbance_data: Absorbance matrix [n_concs, n_wavelengths]
            device: Computation device
            smoothing_factor: Spline smoothing parameter for noise reduction
        """
        self.concentrations = concentrations
        self.wavelengths = wavelengths
        self.absorbance_data = absorbance_data
        self.device = device
        self.smoothing_factor = smoothing_factor
        
        # Normalize concentration range to [-1, 1] for consistency with model
        self.c_min, self.c_max = concentrations.min(), concentrations.max()
        self.c_normalized = 2 * (concentrations - self.c_min) / (self.c_max - self.c_min) - 1
        
        # Normalize wavelength range to [-1, 1] 
        self.wl_min, self.wl_max = wavelengths.min(), wavelengths.max()
        self.wl_normalized = 2 * (wavelengths - self.wl_min) / (self.wl_max - self.wl_min) - 1
        
        # Pre-compute target Christoffel symbols for all wavelengths
        self.target_christoffel_grid = {}
        self.fitted_splines = {}
        self._precompute_targets()
        
    def _precompute_targets(self):
        """Pre-compute target Christoffel symbols for all wavelengths"""
        print("🎯 Computing target Christoffel symbols from spectral data...")
        start_time = time.time()
        
        for wl_idx, wavelength in enumerate(self.wavelengths):
            # Get absorbance curve for this wavelength
            absorbance_curve = self.absorbance_data[:, wl_idx]
            
            # Compute target Christoffel for this wavelength
            c_dense, christoffel_target = self._compute_single_wavelength_target(
                self.c_normalized, absorbance_curve, wavelength
            )
            
            # Store results
            wl_norm = self.wl_normalized[wl_idx]
            self.target_christoffel_grid[float(wl_norm)] = {
                'c_points': c_dense,
                'christoffel': christoffel_target,
                'wavelength_idx': wl_idx
            }
            
        compute_time = time.time() - start_time
        print(f"✅ Target Christoffel computation completed in {compute_time:.2f}s")
        print(f"   Computed targets for {len(self.wavelengths)} wavelengths")
        
    def _compute_single_wavelength_target(self, 
                                        c_points: np.ndarray, 
                                        absorbance_values: np.ndarray,
                                        wavelength: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute target Christoffel symbols for a single wavelength
        
        The geodesic equation is: d²c/dt² + Γ(c,λ) * (dc/dt)² = 0
        If we parameterize the spectral curve as c(t), A(t), then:
        Γ(c,λ) = -d²c/dt² / (dc/dt)²
        
        Args:
            c_points: Concentration points (normalized)
            absorbance_values: Absorbance values at these concentrations
            wavelength: Current wavelength
            
        Returns:
            c_dense: Dense concentration grid
            christoffel_target: Target Christoffel values
        """
        # Remove any NaN or infinite values
        valid_mask = np.isfinite(absorbance_values) & np.isfinite(c_points)
        if not np.all(valid_mask):
            c_points = c_points[valid_mask]
            absorbance_values = absorbance_values[valid_mask]
            
        if len(c_points) < 4:
            # Not enough points for spline fitting - return zeros
            c_dense = np.linspace(-1, 1, 100)
            return c_dense, np.zeros_like(c_dense)
        
        # Smooth the data to reduce noise effects
        if self.smoothing_factor > 0:
            absorbance_values = gaussian_filter1d(absorbance_values, sigma=0.5)
        
        # Fit smooth spline to the data
        try:
            # Use UnivariateSpline for automatic smoothing
            spline = UnivariateSpline(c_points, absorbance_values, 
                                    s=self.smoothing_factor * len(c_points))
            self.fitted_splines[wavelength] = spline
            
        except Exception as e:
            print(f"⚠️  Spline fitting failed for λ={wavelength:.1f}nm: {e}")
            # Fallback to linear interpolation
            c_dense = np.linspace(-1, 1, 100) 
            return c_dense, np.zeros_like(c_dense)
        
        # Create dense concentration grid for target computation
        c_dense = np.linspace(-1, 1, 200)  # High resolution for smooth derivatives
        
        # Parameterize the curve using arc-length-like parameter
        # For each c value, we have corresponding A value from spline
        A_dense = spline(c_dense)
        
        # Compute derivatives using finite differences with high accuracy
        # Use central differences for better accuracy
        dc = c_dense[1] - c_dense[0]
        
        # First derivative dc/dt (normalized by parameter step)
        dc_dt = np.gradient(c_dense, dc)
        
        # Second derivative d²c/dt²  
        d2c_dt2 = np.gradient(dc_dt, dc)
        
        # Compute target Christoffel: Γ = -d²c/dt² / (dc/dt)²
        # Add regularization to prevent division by zero
        eps = 1e-6
        denominator = dc_dt**2 + eps
        christoffel_target = -d2c_dt2 / denominator
        
        # Apply bounds to prevent extreme values
        christoffel_max = 50.0  # Reasonable upper bound
        christoffel_target = np.clip(christoffel_target, -christoffel_max, christoffel_max)
        
        # Additional smoothing of the target Christoffel
        christoffel_target = gaussian_filter1d(christoffel_target, sigma=1.0)
        
        return c_dense, christoffel_target
    
    def get_target_christoffel(self, c_norm: torch.Tensor, wl_norm: torch.Tensor) -> torch.Tensor:
        """
        Get target Christoffel symbols for given concentration/wavelength pairs
        
        Args:
            c_norm: Normalized concentration values [-1, 1] [batch_size]
            wl_norm: Normalized wavelength values [-1, 1] [batch_size] 
            
        Returns:
            Target Christoffel symbols [batch_size]
        """
        batch_size = c_norm.shape[0]
        targets = torch.zeros(batch_size, device=self.device)
        
        for i in range(batch_size):
            c_val = float(c_norm[i])
            wl_val = float(wl_norm[i])
            
            # Find closest wavelength in our pre-computed grid
            closest_wl = min(self.target_christoffel_grid.keys(), 
                           key=lambda x: abs(x - wl_val))
            
            # Get pre-computed data for this wavelength
            wl_data = self.target_christoffel_grid[closest_wl]
            c_points = wl_data['c_points']
            christoffel_values = wl_data['christoffel']
            
            # Interpolate to get target at desired concentration
            target_val = np.interp(c_val, c_points, christoffel_values)
            targets[i] = float(target_val)
            
        return targets
    
    def get_interpolated_absorbance(self, c_norm: torch.Tensor, wl_norm: torch.Tensor) -> torch.Tensor:
        """
        Get interpolated absorbance values using fitted splines
        Useful for validation and testing
        
        Args:
            c_norm: Normalized concentrations [-1, 1] [batch_size]
            wl_norm: Normalized wavelengths [-1, 1] [batch_size]
            
        Returns:
            Interpolated absorbance values [batch_size]
        """
        batch_size = c_norm.shape[0]
        absorbances = torch.zeros(batch_size, device=self.device)
        
        for i in range(batch_size):
            c_val = float(c_norm[i])
            wl_val = float(wl_norm[i])
            
            # Convert back to original wavelength scale
            wl_orig = (wl_val + 1) * (self.wl_max - self.wl_min) / 2 + self.wl_min
            
            # Find closest fitted spline
            if wl_orig in self.fitted_splines:
                spline = self.fitted_splines[wl_orig]
                abs_val = spline(c_val)
            else:
                # Find closest wavelength 
                closest_wl = min(self.fitted_splines.keys(),
                               key=lambda x: abs(x - wl_orig))
                spline = self.fitted_splines[closest_wl]
                abs_val = spline(c_val)
                
            absorbances[i] = float(abs_val)
            
        return absorbances
    
    def validate_targets(self) -> Dict[str, float]:
        """
        Validate the computed target Christoffel symbols
        
        Returns:
            Dictionary with validation metrics
        """
        print("🔍 Validating target Christoffel symbols...")
        
        metrics = {
            'n_wavelengths': len(self.target_christoffel_grid),
            'mean_christoffel': 0.0,
            'std_christoffel': 0.0,
            'max_christoffel': 0.0,
            'min_christoffel': 0.0,
            'nan_count': 0,
            'inf_count': 0
        }
        
        all_christoffel_values = []
        nan_count = 0
        inf_count = 0
        
        for wl_norm, data in self.target_christoffel_grid.items():
            christoffel_vals = data['christoffel']
            all_christoffel_values.extend(christoffel_vals)
            
            # Count problematic values
            nan_count += np.isnan(christoffel_vals).sum()
            inf_count += np.isinf(christoffel_vals).sum()
        
        all_christoffel_values = np.array(all_christoffel_values)
        valid_values = all_christoffel_values[np.isfinite(all_christoffel_values)]
        
        if len(valid_values) > 0:
            metrics.update({
                'mean_christoffel': float(np.mean(valid_values)),
                'std_christoffel': float(np.std(valid_values)),
                'max_christoffel': float(np.max(valid_values)),
                'min_christoffel': float(np.min(valid_values)),
                'nan_count': int(nan_count),
                'inf_count': int(inf_count)
            })
        
        print(f"   Mean Christoffel: {metrics['mean_christoffel']:.4f}")
        print(f"   Std Christoffel: {metrics['std_christoffel']:.4f}")
        print(f"   Range: [{metrics['min_christoffel']:.4f}, {metrics['max_christoffel']:.4f}]")
        print(f"   Problematic values: {nan_count} NaN, {inf_count} Inf")
        
        return metrics
    
    def plot_target_examples(self, n_examples: int = 4):
        """
        Plot examples of fitted curves and target Christoffel symbols
        """
        import matplotlib.pyplot as plt
        
        # Select wavelengths for plotting
        wl_keys = list(self.target_christoffel_grid.keys())
        selected_wls = wl_keys[::len(wl_keys)//n_examples][:n_examples]
        
        fig, axes = plt.subplots(2, n_examples, figsize=(16, 8))
        if n_examples == 1:
            axes = axes.reshape(2, 1)
        
        for i, wl_norm in enumerate(selected_wls):
            data = self.target_christoffel_grid[wl_norm]
            c_points = data['c_points']
            christoffel = data['christoffel']
            wl_idx = data['wavelength_idx']
            wl_orig = self.wavelengths[wl_idx]
            
            # Plot original data and fitted curve
            axes[0, i].scatter(self.c_normalized, self.absorbance_data[:, wl_idx], 
                             alpha=0.7, label='Data', s=30)
            if wl_orig in self.fitted_splines:
                spline = self.fitted_splines[wl_orig]
                c_fine = np.linspace(-1, 1, 200)
                a_fine = spline(c_fine)
                axes[0, i].plot(c_fine, a_fine, 'r-', label='Fitted Spline', linewidth=2)
            
            axes[0, i].set_title(f'λ = {wl_orig:.0f} nm')
            axes[0, i].set_xlabel('Concentration (norm)')
            axes[0, i].set_ylabel('Absorbance')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # Plot target Christoffel symbols
            axes[1, i].plot(c_points, christoffel, 'b-', linewidth=2)
            axes[1, i].set_xlabel('Concentration (norm)')
            axes[1, i].set_ylabel('Target Christoffel Γ')
            axes[1, i].grid(True, alpha=0.3)
            axes[1, i].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('outputs/target_christoffel_examples.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 Target Christoffel examples saved to outputs/target_christoffel_examples.png")