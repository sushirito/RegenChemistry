"""
Data adherence tests - verify model predictions align with known data points
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys
import subprocess

# Add parent directories to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'geodesic_m1'))

from models.geodesic_model import GeodesicNODE
from training.data_loader import SpectralDataset


class DataAdherenceTests:
    """Test suite for verifying model adheres to known spectral data"""
    
    def __init__(self, model: GeodesicNODE, dataset: SpectralDataset, tolerance: float = 0.02):
        """
        Initialize data adherence tests
        
        Args:
            model: Trained geodesic NODE model
            dataset: Dataset with known spectral data
            tolerance: Acceptable error tolerance (default 2%)
        """
        self.model = model
        self.dataset = dataset
        self.tolerance = tolerance
        self.device = model.device
        
        # Get data arrays from dataset
        self.concentrations = dataset.concentration_values
        self.wavelengths = dataset.wavelengths
        self.absorbance_data = dataset.absorbance_data
        
        # Track test results
        self.test_results = {}
        
    def test_known_concentration_accuracy(self, verbose: bool = True) -> Dict[str, float]:
        """
        Test that model predicts exact values at known concentrations
        
        Returns:
            Dictionary with accuracy metrics
        """
        if verbose:
            print("🧪 Testing known concentration accuracy...")
        
        errors = []
        predictions = []
        targets = []
        test_points = []
        
        # Test all known concentration-wavelength pairs
        for i, c_true in enumerate(self.concentrations):
            for j, wl_true in enumerate(self.wavelengths):
                # Normalize inputs to [-1, 1] range as expected by model
                c_norm = 2 * (c_true - self.concentrations.min()) / (self.concentrations.max() - self.concentrations.min()) - 1
                wl_norm = 2 * (wl_true - self.wavelengths.min()) / (self.wavelengths.max() - self.wavelengths.min()) - 1
                
                # Get true absorbance value
                true_absorbance = self.absorbance_data[i, j]
                
                # Predict using model (self-transition: same source and target)
                with torch.no_grad():
                    c_tensor = torch.tensor([c_norm], device=self.device, dtype=torch.float32)
                    wl_tensor = torch.tensor([wl_norm], device=self.device, dtype=torch.float32)
                    
                    result = self.model.forward(c_tensor, c_tensor, wl_tensor)
                    pred_absorbance = float(result['absorbance'][0])
                
                # Calculate error
                error = abs(pred_absorbance - true_absorbance)
                relative_error = error / (abs(true_absorbance) + 1e-8)
                
                errors.append(error)
                predictions.append(pred_absorbance)
                targets.append(true_absorbance)
                test_points.append((c_true, wl_true, error))
        
        # Calculate metrics
        errors = np.array(errors)
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        results = {
            'mean_absolute_error': float(np.mean(errors)),
            'max_error': float(np.max(errors)),
            'rmse': float(np.sqrt(np.mean(errors**2))),
            'r2_score': float(1 - np.sum(errors**2) / np.sum((targets - np.mean(targets))**2)),
            'fraction_within_tolerance': float(np.mean(errors < self.tolerance)),
            'n_test_points': len(errors)
        }
        
        # Find worst performing points
        worst_indices = np.argsort(errors)[-5:]  # 5 worst points
        worst_points = [test_points[i] for i in worst_indices]
        
        if verbose:
            print(f"   MAE: {results['mean_absolute_error']:.4f}")
            print(f"   Max Error: {results['max_error']:.4f}")
            print(f"   RMSE: {results['rmse']:.4f}")
            print(f"   R²: {results['r2_score']:.4f}")
            print(f"   Within Tolerance ({self.tolerance}): {results['fraction_within_tolerance']:.1%}")
            print(f"   Worst points: {[f'c={c:.0f}, λ={w:.0f}, err={e:.4f}' for c,w,e in worst_points[:3]]}")
        
        # Store for later analysis
        self.test_results['known_concentration_accuracy'] = results
        self.test_results['known_concentration_errors'] = errors
        self.test_results['known_concentration_predictions'] = predictions
        self.test_results['known_concentration_targets'] = targets
        
        return results
    
    def test_interpolation_smoothness(self, n_test_wavelengths: int = 10, n_points: int = 100, verbose: bool = True) -> Dict[str, float]:
        """
        Test that interpolations between known concentrations are smooth
        
        Args:
            n_test_wavelengths: Number of wavelengths to test
            n_points: Number of interpolation points to test
            
        Returns:
            Smoothness metrics
        """
        if verbose:
            print("🧪 Testing interpolation smoothness...")
        
        # Select test wavelengths
        wl_indices = np.linspace(0, len(self.wavelengths)-1, n_test_wavelengths, dtype=int)
        test_wavelengths = self.wavelengths[wl_indices]
        
        smoothness_scores = []
        monotonicity_violations = []
        
        for wl in test_wavelengths:
            # Create dense concentration grid
            c_min, c_max = self.concentrations.min(), self.concentrations.max()
            c_dense = np.linspace(c_min, c_max, n_points)
            c_norm_dense = 2 * (c_dense - c_min) / (c_max - c_min) - 1
            wl_norm = 2 * (wl - self.wavelengths.min()) / (self.wavelengths.max() - self.wavelengths.min()) - 1
            
            # Get predictions along this wavelength
            predictions = []
            for c_norm in c_norm_dense:
                with torch.no_grad():
                    c_tensor = torch.tensor([c_norm], device=self.device, dtype=torch.float32)
                    wl_tensor = torch.tensor([wl_norm], device=self.device, dtype=torch.float32)
                    
                    result = self.model.forward(c_tensor, c_tensor, wl_tensor)
                    pred = float(result['absorbance'][0])
                    predictions.append(pred)
            
            predictions = np.array(predictions)
            
            # Test smoothness using second derivative
            second_derivative = np.gradient(np.gradient(predictions))
            smoothness = 1.0 / (1.0 + np.mean(second_derivative**2))
            smoothness_scores.append(smoothness)
            
            # Test for oscillations (rapid sign changes in derivative)
            first_derivative = np.gradient(predictions)
            sign_changes = np.sum(np.diff(np.sign(first_derivative)) != 0)
            monotonicity_violations.append(sign_changes)
        
        results = {
            'mean_smoothness_score': float(np.mean(smoothness_scores)),
            'min_smoothness_score': float(np.min(smoothness_scores)),
            'mean_monotonicity_violations': float(np.mean(monotonicity_violations)),
            'max_monotonicity_violations': int(np.max(monotonicity_violations)),
            'n_wavelengths_tested': len(test_wavelengths)
        }
        
        if verbose:
            print(f"   Mean smoothness: {results['mean_smoothness_score']:.4f}")
            print(f"   Min smoothness: {results['min_smoothness_score']:.4f}")
            print(f"   Mean monotonicity violations: {results['mean_monotonicity_violations']:.1f}")
            print(f"   Max monotonicity violations: {results['max_monotonicity_violations']}")
        
        self.test_results['interpolation_smoothness'] = results
        return results
    
    def test_spectral_curve_shape(self, test_concentrations: List[float] = None, verbose: bool = True) -> Dict[str, float]:
        """
        Test that spectral curves maintain proper shape across wavelengths
        
        Args:
            test_concentrations: Concentrations to test (if None, use subset of known values)
            
        Returns:
            Shape validation metrics
        """
        if verbose:
            print("🧪 Testing spectral curve shapes...")
        
        if test_concentrations is None:
            # Use every other known concentration
            test_concentrations = self.concentrations[::2]
        
        shape_scores = []
        peak_detection_scores = []
        
        for c in test_concentrations:
            # Normalize concentration
            c_norm = 2 * (c - self.concentrations.min()) / (self.concentrations.max() - self.concentrations.min()) - 1
            
            # Get predictions across all wavelengths
            predictions = []
            for wl in self.wavelengths:
                wl_norm = 2 * (wl - self.wavelengths.min()) / (self.wavelengths.max() - self.wavelengths.min()) - 1
                
                with torch.no_grad():
                    c_tensor = torch.tensor([c_norm], device=self.device, dtype=torch.float32)
                    wl_tensor = torch.tensor([wl_norm], device=self.device, dtype=torch.float32)
                    
                    result = self.model.forward(c_tensor, c_tensor, wl_tensor)
                    pred = float(result['absorbance'][0])
                    predictions.append(pred)
            
            predictions = np.array(predictions)
            
            # Compare with true spectral curve at nearest known concentration
            nearest_c_idx = np.argmin(np.abs(self.concentrations - c))
            true_curve = self.absorbance_data[nearest_c_idx, :]
            
            # Shape similarity (correlation)
            if len(predictions) > 1 and len(true_curve) > 1:
                shape_correlation = np.corrcoef(predictions, true_curve)[0, 1]
                if np.isnan(shape_correlation):
                    shape_correlation = 0.0
                shape_scores.append(shape_correlation)
            
            # Peak detection consistency
            pred_peaks = self._find_peaks(predictions)
            true_peaks = self._find_peaks(true_curve)
            peak_similarity = self._compare_peaks(pred_peaks, true_peaks)
            peak_detection_scores.append(peak_similarity)
        
        results = {
            'mean_shape_correlation': float(np.mean(shape_scores)),
            'min_shape_correlation': float(np.min(shape_scores)),
            'mean_peak_similarity': float(np.mean(peak_detection_scores)),
            'min_peak_similarity': float(np.min(peak_detection_scores)),
            'n_concentrations_tested': len(test_concentrations)
        }
        
        if verbose:
            print(f"   Mean shape correlation: {results['mean_shape_correlation']:.4f}")
            print(f"   Min shape correlation: {results['min_shape_correlation']:.4f}")
            print(f"   Mean peak similarity: {results['mean_peak_similarity']:.4f}")
        
        self.test_results['spectral_curve_shape'] = results
        return results
    
    def test_concentration_boundary_behavior(self, verbose: bool = True) -> Dict[str, float]:
        """
        Test behavior at concentration boundaries (0 and max concentration)
        
        Returns:
            Boundary behavior metrics
        """
        if verbose:
            print("🧪 Testing concentration boundary behavior...")
        
        c_min = self.concentrations.min()
        c_max = self.concentrations.max()
        
        # Test wavelengths (subset for efficiency)
        test_wl_indices = np.linspace(0, len(self.wavelengths)-1, 20, dtype=int)
        test_wavelengths = self.wavelengths[test_wl_indices]
        
        boundary_errors = {'min': [], 'max': []}
        extrapolation_stability = []
        
        for wl in test_wavelengths:
            wl_norm = 2 * (wl - self.wavelengths.min()) / (self.wavelengths.max() - self.wavelengths.min()) - 1
            
            # Test at boundaries
            for boundary, c_val in [('min', c_min), ('max', c_max)]:
                c_norm = 2 * (c_val - c_min) / (c_max - c_min) - 1
                
                # Get nearest known data point
                nearest_c_idx = 0 if boundary == 'min' else -1
                nearest_wl_idx = np.argmin(np.abs(self.wavelengths - wl))
                true_val = self.absorbance_data[nearest_c_idx, nearest_wl_idx]
                
                # Predict at boundary
                with torch.no_grad():
                    c_tensor = torch.tensor([c_norm], device=self.device, dtype=torch.float32)
                    wl_tensor = torch.tensor([wl_norm], device=self.device, dtype=torch.float32)
                    
                    result = self.model.forward(c_tensor, c_tensor, wl_tensor)
                    pred_val = float(result['absorbance'][0])
                
                error = abs(pred_val - true_val)
                boundary_errors[boundary].append(error)
            
            # Test extrapolation stability (slightly beyond boundaries)
            for delta in [-0.1, 0.1]:  # Slightly outside [-1, 1] range
                c_extrap = -1.0 + delta if delta < 0 else 1.0 + delta
                c_extrap = max(-1.2, min(1.2, c_extrap))  # Clamp to reasonable range
                
                with torch.no_grad():
                    c_tensor = torch.tensor([c_extrap], device=self.device, dtype=torch.float32)
                    wl_tensor = torch.tensor([wl_norm], device=self.device, dtype=torch.float32)
                    
                    try:
                        result = self.model.forward(c_tensor, c_tensor, wl_tensor)
                        pred_val = float(result['absorbance'][0])
                        # Check if prediction is reasonable (not NaN, Inf, or extreme)
                        if np.isfinite(pred_val) and abs(pred_val) < 10:
                            extrapolation_stability.append(1.0)
                        else:
                            extrapolation_stability.append(0.0)
                    except:
                        extrapolation_stability.append(0.0)
        
        results = {
            'boundary_error_min': float(np.mean(boundary_errors['min'])),
            'boundary_error_max': float(np.mean(boundary_errors['max'])),
            'max_boundary_error': float(max(np.max(boundary_errors['min']), np.max(boundary_errors['max']))),
            'extrapolation_stability': float(np.mean(extrapolation_stability)),
            'n_wavelengths_tested': len(test_wavelengths)
        }
        
        if verbose:
            print(f"   Min boundary error: {results['boundary_error_min']:.4f}")
            print(f"   Max boundary error: {results['boundary_error_max']:.4f}")
            print(f"   Extrapolation stability: {results['extrapolation_stability']:.1%}")
        
        self.test_results['concentration_boundary_behavior'] = results
        return results
    
    def run_all_data_adherence_tests(self, verbose: bool = True) -> Dict[str, any]:
        """
        Run all data adherence tests and return comprehensive results
        
        Returns:
            Complete test results dictionary
        """
        if verbose:
            print("🔬 Running complete data adherence test suite...\n")
        
        # Run all tests
        accuracy_results = self.test_known_concentration_accuracy(verbose)
        smoothness_results = self.test_interpolation_smoothness(verbose)
        shape_results = self.test_spectral_curve_shape(verbose)
        boundary_results = self.test_concentration_boundary_behavior(verbose)
        
        # Compute overall score
        overall_score = self._compute_overall_score()
        
        # Create comprehensive results
        comprehensive_results = {
            'overall_score': overall_score,
            'accuracy': accuracy_results,
            'smoothness': smoothness_results,
            'shape': shape_results,
            'boundary': boundary_results,
            'pass_fail_summary': self._generate_pass_fail_summary()
        }
        
        if verbose:
            print(f"\n📊 Overall Data Adherence Score: {overall_score:.2f}/100")
            self._print_summary(comprehensive_results)
        
        return comprehensive_results
    
    def _find_peaks(self, curve: np.ndarray) -> List[int]:
        """Simple peak detection"""
        peaks = []
        for i in range(1, len(curve) - 1):
            if curve[i] > curve[i-1] and curve[i] > curve[i+1]:
                peaks.append(i)
        return peaks
    
    def _compare_peaks(self, peaks1: List[int], peaks2: List[int]) -> float:
        """Compare peak locations between two curves"""
        if len(peaks1) == 0 and len(peaks2) == 0:
            return 1.0
        if len(peaks1) == 0 or len(peaks2) == 0:
            return 0.0
        
        # Simple peak similarity based on count and approximate locations
        count_similarity = min(len(peaks1), len(peaks2)) / max(len(peaks1), len(peaks2))
        
        # Location similarity (simplified)
        if len(peaks1) > 0 and len(peaks2) > 0:
            location_similarity = 1.0 - abs(np.mean(peaks1) - np.mean(peaks2)) / len(peaks1)
            location_similarity = max(0, location_similarity)
        else:
            location_similarity = 0.0
        
        return 0.5 * (count_similarity + location_similarity)
    
    def _compute_overall_score(self) -> float:
        """Compute overall data adherence score (0-100)"""
        if not self.test_results:
            return 0.0
        
        scores = []
        weights = []
        
        # Accuracy score (most important - 40%)
        if 'known_concentration_accuracy' in self.test_results:
            r2 = self.test_results['known_concentration_accuracy']['r2_score']
            accuracy_score = max(0, min(100, r2 * 100))
            scores.append(accuracy_score)
            weights.append(0.4)
        
        # Smoothness score (25%)
        if 'interpolation_smoothness' in self.test_results:
            smoothness = self.test_results['interpolation_smoothness']['mean_smoothness_score']
            smoothness_score = smoothness * 100
            scores.append(smoothness_score)
            weights.append(0.25)
        
        # Shape score (25%)
        if 'spectral_curve_shape' in self.test_results:
            shape_corr = self.test_results['spectral_curve_shape']['mean_shape_correlation']
            shape_score = max(0, shape_corr * 100)
            scores.append(shape_score)
            weights.append(0.25)
        
        # Boundary score (10%)
        if 'concentration_boundary_behavior' in self.test_results:
            stability = self.test_results['concentration_boundary_behavior']['extrapolation_stability']
            boundary_score = stability * 100
            scores.append(boundary_score)
            weights.append(0.1)
        
        if scores:
            overall = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            return float(overall)
        else:
            return 0.0
    
    def _generate_pass_fail_summary(self) -> Dict[str, bool]:
        """Generate pass/fail summary for each test category"""
        summary = {}
        
        # Define pass thresholds
        thresholds = {
            'accuracy_r2': 0.8,  # R² > 0.8
            'accuracy_mae': 0.05,  # MAE < 0.05
            'smoothness': 0.7,  # Smoothness > 0.7
            'shape_correlation': 0.6,  # Shape correlation > 0.6
            'extrapolation_stability': 0.8  # Stability > 80%
        }
        
        # Check accuracy
        if 'known_concentration_accuracy' in self.test_results:
            acc = self.test_results['known_concentration_accuracy']
            summary['accuracy_pass'] = (acc['r2_score'] >= thresholds['accuracy_r2'] and 
                                      acc['mean_absolute_error'] <= thresholds['accuracy_mae'])
        
        # Check smoothness
        if 'interpolation_smoothness' in self.test_results:
            smooth = self.test_results['interpolation_smoothness']
            summary['smoothness_pass'] = smooth['mean_smoothness_score'] >= thresholds['smoothness']
        
        # Check shape
        if 'spectral_curve_shape' in self.test_results:
            shape = self.test_results['spectral_curve_shape']
            summary['shape_pass'] = shape['mean_shape_correlation'] >= thresholds['shape_correlation']
        
        # Check boundary behavior
        if 'concentration_boundary_behavior' in self.test_results:
            boundary = self.test_results['concentration_boundary_behavior']
            summary['boundary_pass'] = boundary['extrapolation_stability'] >= thresholds['extrapolation_stability']
        
        # Overall pass
        summary['overall_pass'] = all(summary.values()) if summary else False
        
        return summary
    
    def _print_summary(self, results: Dict[str, any]):
        """Print formatted test summary"""
        print("\n" + "="*60)
        print("DATA ADHERENCE TEST SUMMARY")
        print("="*60)
        
        summary = results['pass_fail_summary']
        
        def status_emoji(passed):
            return "✅" if passed else "❌"
        
        print(f"{status_emoji(summary.get('accuracy_pass', False))} Accuracy Test: R² = {results['accuracy']['r2_score']:.3f}")
        print(f"{status_emoji(summary.get('smoothness_pass', False))} Smoothness Test: Score = {results['smoothness']['mean_smoothness_score']:.3f}")
        print(f"{status_emoji(summary.get('shape_pass', False))} Shape Test: Correlation = {results['shape']['mean_shape_correlation']:.3f}")
        print(f"{status_emoji(summary.get('boundary_pass', False))} Boundary Test: Stability = {results['boundary']['extrapolation_stability']:.1%}")
        print(f"\n{status_emoji(summary.get('overall_pass', False))} OVERALL: {'PASS' if summary.get('overall_pass', False) else 'FAIL'}")
        print("="*60)
    
    def create_adherence_visualization(self, save_path: str = "outputs/data_adherence_results.png"):
        """
        Create comprehensive visualization of data adherence test results
        """
        if not self.test_results:
            print("❌ No test results available for visualization")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Data Adherence Test Results', fontsize=16, fontweight='bold')
        
        # 1. Accuracy scatter plot
        if 'known_concentration_errors' in self.test_results:
            predictions = self.test_results['known_concentration_predictions']
            targets = self.test_results['known_concentration_targets']
            
            axes[0, 0].scatter(targets, predictions, alpha=0.6, s=10)
            axes[0, 0].plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--', alpha=0.8)
            axes[0, 0].set_xlabel('True Absorbance')
            axes[0, 0].set_ylabel('Predicted Absorbance')
            axes[0, 0].set_title('Prediction vs Truth')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add R² annotation
            r2 = self.test_results['known_concentration_accuracy']['r2_score']
            axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0, 0].transAxes, 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Error distribution
        if 'known_concentration_errors' in self.test_results:
            errors = self.test_results['known_concentration_errors']
            axes[0, 1].hist(errors, bins=30, alpha=0.7, edgecolor='black')
            axes[0, 1].axvline(self.tolerance, color='red', linestyle='--', label=f'Tolerance: {self.tolerance}')
            axes[0, 1].set_xlabel('Absolute Error')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Error Distribution')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Sample spectral curves comparison
        if hasattr(self, 'concentrations') and hasattr(self, 'wavelengths'):
            # Show a few example spectral curves
            test_c_indices = [0, len(self.concentrations)//2, -1]  # First, middle, last
            for i, c_idx in enumerate(test_c_indices):
                c_val = self.concentrations[c_idx]
                c_norm = 2 * (c_val - self.concentrations.min()) / (self.concentrations.max() - self.concentrations.min()) - 1
                
                # Get predictions for this concentration
                predictions = []
                for wl in self.wavelengths:
                    wl_norm = 2 * (wl - self.wavelengths.min()) / (self.wavelengths.max() - self.wavelengths.min()) - 1
                    
                    with torch.no_grad():
                        c_tensor = torch.tensor([c_norm], device=self.device, dtype=torch.float32)
                        wl_tensor = torch.tensor([wl_norm], device=self.device, dtype=torch.float32)
                        
                        result = self.model.forward(c_tensor, c_tensor, wl_tensor)
                        pred = float(result['absorbance'][0])
                        predictions.append(pred)
                
                # Plot true vs predicted
                true_curve = self.absorbance_data[c_idx, :]
                color = plt.cm.viridis(i / max(1, len(test_c_indices)-1))
                
                axes[0, 2].plot(self.wavelengths, true_curve, 'o-', color=color, alpha=0.7, 
                               label=f'True {c_val:.0f} ppb', markersize=3)
                axes[0, 2].plot(self.wavelengths, predictions, '--', color=color, alpha=0.9,
                               label=f'Pred {c_val:.0f} ppb', linewidth=2)
            
            axes[0, 2].set_xlabel('Wavelength (nm)')
            axes[0, 2].set_ylabel('Absorbance')
            axes[0, 2].set_title('Sample Spectral Curves')
            axes[0, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4-6: Test score summary as bar charts
        test_categories = ['Accuracy', 'Smoothness', 'Shape', 'Boundary']
        test_scores = []
        test_colors = []
        
        if 'known_concentration_accuracy' in self.test_results:
            score = max(0, min(100, self.test_results['known_concentration_accuracy']['r2_score'] * 100))
            test_scores.append(score)
            test_colors.append('green' if score >= 80 else 'red')
        
        if 'interpolation_smoothness' in self.test_results:
            score = self.test_results['interpolation_smoothness']['mean_smoothness_score'] * 100
            test_scores.append(score)
            test_colors.append('green' if score >= 70 else 'red')
        
        if 'spectral_curve_shape' in self.test_results:
            score = max(0, self.test_results['spectral_curve_shape']['mean_shape_correlation'] * 100)
            test_scores.append(score)
            test_colors.append('green' if score >= 60 else 'red')
        
        if 'concentration_boundary_behavior' in self.test_results:
            score = self.test_results['concentration_boundary_behavior']['extrapolation_stability'] * 100
            test_scores.append(score)
            test_colors.append('green' if score >= 80 else 'red')
        
        # Plot test scores
        if test_scores:
            bars = axes[1, 0].bar(test_categories[:len(test_scores)], test_scores, color=test_colors, alpha=0.7)
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].set_title('Test Category Scores')
            axes[1, 0].set_ylim(0, 100)
            axes[1, 0].axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='Pass Threshold')
            
            # Add score labels on bars
            for bar, score in zip(bars, test_scores):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                               f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
            
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(1, 3):
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Data adherence visualization saved to {save_path}")
        
        # Show plot
        try:
            plt.show()
            # Also try to open automatically
            subprocess.run(['open', save_path], check=False)
        except:
            pass
        
        return save_path