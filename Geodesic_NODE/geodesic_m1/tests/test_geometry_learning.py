"""
Geometry learning tests - verify Christoffel symbols and geodesic paths match targets
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


class GeometryLearningTests:
    """Test suite for validating geometry learning and Christoffel symbol matching"""
    
    def __init__(self, model: GeodesicNODE, dataset: SpectralDataset):
        """
        Initialize geometry learning tests
        
        Args:
            model: Trained geodesic NODE model
            dataset: Dataset with spectral data
        """
        self.model = model
        self.dataset = dataset
        self.device = model.device
        
        # Get data arrays
        self.concentrations = dataset.concentration_values
        self.wavelengths = dataset.wavelengths
        self.absorbance_data = dataset.absorbance_data
        
        # Track test results
        self.test_results = {}
        
    def test_christoffel_target_matching(self, n_test_points: int = 100, verbose: bool = True) -> Dict[str, float]:
        """
        Test how well learned Christoffel symbols match computed targets
        
        Args:
            n_test_points: Number of random points to test
            
        Returns:
            Christoffel matching metrics
        """
        if verbose:
            print("🧪 Testing Christoffel target matching...")
        
        if self.model.target_christoffel_computer is None:
            if verbose:
                print("   ⚠️  No target Christoffel computer available")
            return {'error': 'No target data available'}
        
        # Generate random test points in normalized [-1, 1] space
        c_test = torch.rand(n_test_points, device=self.device) * 2 - 1  # [-1, 1]
        wl_test = torch.rand(n_test_points, device=self.device) * 2 - 1  # [-1, 1]
        
        # Get target and actual Christoffel symbols
        with torch.no_grad():
            target_christoffel = self.model.target_christoffel_computer.get_target_christoffel(c_test, wl_test)
            actual_christoffel = self.model.christoffel_computer.interpolate(c_test, wl_test)
        
        # Compute metrics
        mse = float(torch.mean((actual_christoffel - target_christoffel)**2))
        mae = float(torch.mean(torch.abs(actual_christoffel - target_christoffel)))
        
        # Correlation coefficient
        if len(actual_christoffel) > 1:
            actual_centered = actual_christoffel - actual_christoffel.mean()
            target_centered = target_christoffel - target_christoffel.mean()
            correlation = float(torch.sum(actual_centered * target_centered) / 
                              (torch.norm(actual_centered) * torch.norm(target_centered) + 1e-8))
        else:
            correlation = 0.0
        
        # R² score
        target_var = torch.var(target_christoffel)
        if target_var > 1e-8:
            r2_score = float(1 - torch.sum((actual_christoffel - target_christoffel)**2) / 
                           (torch.sum((target_christoffel - target_christoffel.mean())**2) + 1e-8))
        else:
            r2_score = 0.0
        
        # Distribution matching
        target_mean = float(target_christoffel.mean())
        target_std = float(target_christoffel.std())
        actual_mean = float(actual_christoffel.mean())
        actual_std = float(actual_christoffel.std())
        
        mean_ratio = actual_mean / (target_mean + 1e-8)
        std_ratio = actual_std / (target_std + 1e-8)
        
        results = {
            'christoffel_mse': mse,
            'christoffel_mae': mae,
            'christoffel_correlation': correlation,
            'christoffel_r2': r2_score,
            'target_mean': target_mean,
            'target_std': target_std,
            'actual_mean': actual_mean,
            'actual_std': actual_std,
            'mean_ratio': mean_ratio,
            'std_ratio': std_ratio,
            'n_test_points': n_test_points
        }
        
        if verbose:
            print(f"   Christoffel MSE: {mse:.6f}")
            print(f"   Christoffel MAE: {mae:.6f}")
            print(f"   Correlation: {correlation:.4f}")
            print(f"   R²: {r2_score:.4f}")
            print(f"   Mean ratio (actual/target): {mean_ratio:.3f}")
            print(f"   Std ratio (actual/target): {std_ratio:.3f}")
        
        self.test_results['christoffel_target_matching'] = results
        return results
    
    def test_metric_curvature_alignment(self, verbose: bool = True) -> Dict[str, float]:
        """
        Test that metric values align with expected spectral curvature
        High curvature regions should have appropriate metric values
        
        Returns:
            Metric-curvature alignment metrics
        """
        if verbose:
            print("🧪 Testing metric-curvature alignment...")
        
        curvature_scores = []
        metric_appropriateness = []
        
        # Test several wavelengths
        test_wl_indices = np.linspace(0, len(self.wavelengths)-1, 10, dtype=int)
        
        for wl_idx in test_wl_indices:
            wl = self.wavelengths[wl_idx]
            wl_norm = 2 * (wl - self.wavelengths.min()) / (self.wavelengths.max() - self.wavelengths.min()) - 1
            
            # Get spectral curve for this wavelength
            spectral_curve = self.absorbance_data[:, wl_idx]
            
            # Compute curvature of spectral curve
            c_points = np.linspace(-1, 1, len(self.concentrations))
            curvature = self._compute_curvature(c_points, spectral_curve)
            
            # Get metric values at same points
            metric_values = []
            for c_norm in c_points:
                with torch.no_grad():
                    c_tensor = torch.tensor([c_norm], device=self.device, dtype=torch.float32)
                    wl_tensor = torch.tensor([wl_norm], device=self.device, dtype=torch.float32)
                    
                    # Get metric value
                    inputs = torch.stack([c_tensor, wl_tensor], dim=1)
                    metric_val = float(self.model.metric_network(inputs)[0])
                    metric_values.append(metric_val)
            
            metric_values = np.array(metric_values)
            
            # Test alignment: high curvature should correlate with specific metric behavior
            # (The exact relationship depends on our metric parameterization)
            if len(curvature) > 1 and len(metric_values) > 1:
                # Compute correlation between curvature and inverse metric
                # (High curvature should correspond to lower metric values for easier travel)
                inv_metric = 1.0 / (metric_values + 1e-8)
                curvature_metric_corr = np.corrcoef(np.abs(curvature), inv_metric)[0, 1]
                if np.isnan(curvature_metric_corr):
                    curvature_metric_corr = 0.0
                
                curvature_scores.append(curvature_metric_corr)
                
                # Test metric smoothness
                metric_smoothness = 1.0 / (1.0 + np.mean(np.gradient(metric_values)**2))
                metric_appropriateness.append(metric_smoothness)
        
        results = {
            'mean_curvature_alignment': float(np.mean(curvature_scores)) if curvature_scores else 0.0,
            'std_curvature_alignment': float(np.std(curvature_scores)) if curvature_scores else 0.0,
            'mean_metric_smoothness': float(np.mean(metric_appropriateness)) if metric_appropriateness else 0.0,
            'n_wavelengths_tested': len(test_wl_indices)
        }
        
        if verbose:
            print(f"   Mean curvature alignment: {results['mean_curvature_alignment']:.4f}")
            print(f"   Mean metric smoothness: {results['mean_metric_smoothness']:.4f}")
        
        self.test_results['metric_curvature_alignment'] = results
        return results
    
    def test_geodesic_path_validation(self, n_test_transitions: int = 20, verbose: bool = True) -> Dict[str, float]:
        """
        Test that geodesic paths follow expected trajectories through spectral space
        
        Args:
            n_test_transitions: Number of concentration transitions to test
            
        Returns:
            Geodesic path validation metrics
        """
        if verbose:
            print("🧪 Testing geodesic path validation...")
        
        path_deviations = []
        convergence_rates = []
        trajectory_smoothness = []
        
        # Test transitions between known concentrations
        for i in range(min(n_test_transitions, len(self.concentrations)-1)):
            c_source = self.concentrations[i]
            c_target = self.concentrations[i+1]
            
            # Normalize concentrations
            c_source_norm = 2 * (c_source - self.concentrations.min()) / (self.concentrations.max() - self.concentrations.min()) - 1
            c_target_norm = 2 * (c_target - self.concentrations.min()) / (self.concentrations.max() - self.concentrations.min()) - 1
            
            # Test several wavelengths for this transition
            test_wl_indices = np.linspace(0, len(self.wavelengths)-1, 5, dtype=int)
            
            for wl_idx in test_wl_indices:
                wl = self.wavelengths[wl_idx]
                wl_norm = 2 * (wl - self.wavelengths.min()) / (self.wavelengths.max() - self.wavelengths.min()) - 1
                
                # Get geodesic trajectory
                with torch.no_grad():
                    c_source_tensor = torch.tensor([c_source_norm], device=self.device, dtype=torch.float32)
                    c_target_tensor = torch.tensor([c_target_norm], device=self.device, dtype=torch.float32)
                    wl_tensor = torch.tensor([wl_norm], device=self.device, dtype=torch.float32)
                    
                    result = self.model.forward(c_source_tensor, c_target_tensor, wl_tensor)
                    
                    if 'trajectories' in result:
                        trajectory = result['trajectories']  # [n_points, batch_size, 3]
                        convergence_mask = result['convergence_mask']
                        
                        # Extract concentration trajectory
                        c_trajectory = trajectory[:, 0, 0].cpu().numpy()  # [n_points]
                        
                        # Test convergence
                        convergence_rates.append(float(convergence_mask[0]))
                        
                        # Test path deviation from straight line
                        t_points = np.linspace(0, 1, len(c_trajectory))
                        linear_path = c_source_norm + t_points * (c_target_norm - c_source_norm)
                        path_deviation = np.mean(np.abs(c_trajectory - linear_path))
                        path_deviations.append(path_deviation)
                        
                        # Test trajectory smoothness
                        if len(c_trajectory) > 2:
                            trajectory_gradient = np.gradient(c_trajectory)
                            smoothness = 1.0 / (1.0 + np.mean(np.gradient(trajectory_gradient)**2))
                            trajectory_smoothness.append(smoothness)
        
        results = {
            'mean_path_deviation': float(np.mean(path_deviations)) if path_deviations else 0.0,
            'std_path_deviation': float(np.std(path_deviations)) if path_deviations else 0.0,
            'mean_convergence_rate': float(np.mean(convergence_rates)) if convergence_rates else 0.0,
            'mean_trajectory_smoothness': float(np.mean(trajectory_smoothness)) if trajectory_smoothness else 0.0,
            'n_trajectories_tested': len(path_deviations)
        }
        
        if verbose:
            print(f"   Mean path deviation: {results['mean_path_deviation']:.4f}")
            print(f"   Mean convergence rate: {results['mean_convergence_rate']:.1%}")
            print(f"   Mean trajectory smoothness: {results['mean_trajectory_smoothness']:.4f}")
        
        self.test_results['geodesic_path_validation'] = results
        return results
    
    def test_flat_vs_curved_regions(self, verbose: bool = True) -> Dict[str, float]:
        """
        Test that model correctly distinguishes flat vs curved spectral regions
        
        Returns:
            Region classification metrics  
        """
        if verbose:
            print("🧪 Testing flat vs curved region detection...")
        
        flat_region_scores = []
        curved_region_scores = []
        classification_accuracy = []
        
        # Analyze each wavelength
        for wl_idx in range(0, len(self.wavelengths), max(1, len(self.wavelengths)//20)):  # Sample every ~5% of wavelengths
            wl = self.wavelengths[wl_idx]
            wl_norm = 2 * (wl - self.wavelengths.min()) / (self.wavelengths.max() - self.wavelengths.min()) - 1
            
            # Get true spectral curve
            true_curve = self.absorbance_data[:, wl_idx]
            c_points = np.linspace(-1, 1, len(self.concentrations))
            
            # Compute true curvature
            true_curvature = self._compute_curvature(c_points, true_curve)
            
            # Get learned Christoffel symbols (proxy for curvature)
            christoffel_values = []
            for c_norm in c_points:
                with torch.no_grad():
                    c_tensor = torch.tensor([c_norm], device=self.device, dtype=torch.float32)
                    wl_tensor = torch.tensor([wl_norm], device=self.device, dtype=torch.float32)
                    
                    christoffel_val = float(self.model.christoffel_computer.interpolate(c_tensor, wl_tensor))
                    christoffel_values.append(christoffel_val)
            
            christoffel_values = np.array(christoffel_values)
            
            # Classify regions as flat or curved based on true curvature
            curvature_threshold = np.percentile(np.abs(true_curvature), 70)  # Top 30% are "curved"
            is_curved_true = np.abs(true_curvature) > curvature_threshold
            
            # Classify based on learned Christoffel symbols
            christoffel_threshold = np.percentile(np.abs(christoffel_values), 70)
            is_curved_learned = np.abs(christoffel_values) > christoffel_threshold
            
            # Compute classification accuracy
            if len(is_curved_true) > 0:
                accuracy = np.mean(is_curved_true == is_curved_learned)
                classification_accuracy.append(accuracy)
            
            # Analyze flat regions (where curvature should be low)
            flat_mask = ~is_curved_true
            if np.any(flat_mask):
                flat_christoffel = christoffel_values[flat_mask]
                flat_score = 1.0 / (1.0 + np.mean(np.abs(flat_christoffel)))  # Lower is better
                flat_region_scores.append(flat_score)
            
            # Analyze curved regions (where curvature should be high)
            curved_mask = is_curved_true
            if np.any(curved_mask):
                curved_christoffel = christoffel_values[curved_mask]
                curved_score = np.mean(np.abs(curved_christoffel))  # Higher is better for curved regions
                curved_region_scores.append(curved_score)
        
        results = {
            'mean_classification_accuracy': float(np.mean(classification_accuracy)) if classification_accuracy else 0.0,
            'mean_flat_region_score': float(np.mean(flat_region_scores)) if flat_region_scores else 0.0,
            'mean_curved_region_score': float(np.mean(curved_region_scores)) if curved_region_scores else 0.0,
            'flat_vs_curved_contrast': 0.0,
            'n_wavelengths_tested': len(flat_region_scores) + len(curved_region_scores)
        }
        
        # Compute contrast between flat and curved regions
        if flat_region_scores and curved_region_scores:
            flat_mean = np.mean(flat_region_scores)
            curved_mean = np.mean(curved_region_scores)
            results['flat_vs_curved_contrast'] = float(curved_mean / (flat_mean + 1e-8))
        
        if verbose:
            print(f"   Classification accuracy: {results['mean_classification_accuracy']:.1%}")
            print(f"   Flat region score: {results['mean_flat_region_score']:.4f}")
            print(f"   Curved region score: {results['mean_curved_region_score']:.4f}")
            print(f"   Flat vs curved contrast: {results['flat_vs_curved_contrast']:.2f}")
        
        self.test_results['flat_vs_curved_regions'] = results
        return results
    
    def run_all_geometry_tests(self, verbose: bool = True) -> Dict[str, any]:
        """
        Run all geometry learning tests and return comprehensive results
        
        Returns:
            Complete geometry test results dictionary
        """
        if verbose:
            print("🔬 Running complete geometry learning test suite...\n")
        
        # Run all tests
        christoffel_results = self.test_christoffel_target_matching(verbose=verbose)
        curvature_results = self.test_metric_curvature_alignment(verbose=verbose)
        path_results = self.test_geodesic_path_validation(verbose=verbose)
        regions_results = self.test_flat_vs_curved_regions(verbose=verbose)
        
        # Compute overall geometry score
        overall_score = self._compute_overall_geometry_score()
        
        # Create comprehensive results
        comprehensive_results = {
            'overall_geometry_score': overall_score,
            'christoffel_matching': christoffel_results,
            'curvature_alignment': curvature_results,
            'path_validation': path_results,
            'region_classification': regions_results,
            'pass_fail_summary': self._generate_geometry_pass_fail_summary()
        }
        
        if verbose:
            print(f"\n📊 Overall Geometry Learning Score: {overall_score:.2f}/100")
            self._print_geometry_summary(comprehensive_results)
        
        return comprehensive_results
    
    def _compute_curvature(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute curvature of a 2D curve using finite differences
        
        Args:
            x: x coordinates
            y: y coordinates
            
        Returns:
            Curvature values
        """
        if len(x) < 3:
            return np.zeros_like(x)
        
        # First and second derivatives
        dx_dt = np.gradient(x)
        dy_dt = np.gradient(y)
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        
        # Curvature formula: κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
        numerator = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2)
        denominator = (dx_dt**2 + dy_dt**2)**(3/2) + 1e-8
        curvature = numerator / denominator
        
        return curvature
    
    def _compute_overall_geometry_score(self) -> float:
        """Compute overall geometry learning score (0-100)"""
        scores = []
        weights = []
        
        # Christoffel matching score (40% - most important)
        if 'christoffel_target_matching' in self.test_results:
            christoffel = self.test_results['christoffel_target_matching']
            if 'christoffel_r2' in christoffel:
                r2 = max(0, min(1, christoffel['christoffel_r2']))
                scores.append(r2 * 100)
                weights.append(0.4)
        
        # Curvature alignment score (25%)
        if 'metric_curvature_alignment' in self.test_results:
            alignment = self.test_results['metric_curvature_alignment']['mean_curvature_alignment']
            alignment_score = max(0, (alignment + 1) * 50)  # Convert [-1,1] to [0,100]
            scores.append(alignment_score)
            weights.append(0.25)
        
        # Path validation score (25%)
        if 'geodesic_path_validation' in self.test_results:
            path = self.test_results['geodesic_path_validation']
            convergence = path['mean_convergence_rate']
            smoothness = path['mean_trajectory_smoothness']
            path_score = (convergence + smoothness) * 50
            scores.append(path_score)
            weights.append(0.25)
        
        # Region classification score (10%)
        if 'flat_vs_curved_regions' in self.test_results:
            regions = self.test_results['flat_vs_curved_regions']
            classification = regions['mean_classification_accuracy']
            region_score = classification * 100
            scores.append(region_score)
            weights.append(0.1)
        
        if scores:
            overall = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            return float(overall)
        else:
            return 0.0
    
    def _generate_geometry_pass_fail_summary(self) -> Dict[str, bool]:
        """Generate pass/fail summary for geometry tests"""
        summary = {}
        
        # Christoffel matching
        if 'christoffel_target_matching' in self.test_results:
            christoffel = self.test_results['christoffel_target_matching']
            r2_pass = christoffel.get('christoffel_r2', 0) >= 0.7
            corr_pass = christoffel.get('christoffel_correlation', 0) >= 0.6
            summary['christoffel_pass'] = r2_pass and corr_pass
        
        # Curvature alignment
        if 'metric_curvature_alignment' in self.test_results:
            alignment = self.test_results['metric_curvature_alignment']['mean_curvature_alignment']
            summary['curvature_pass'] = alignment >= 0.3
        
        # Path validation
        if 'geodesic_path_validation' in self.test_results:
            path = self.test_results['geodesic_path_validation']
            convergence_pass = path['mean_convergence_rate'] >= 0.8
            smoothness_pass = path['mean_trajectory_smoothness'] >= 0.7
            summary['path_pass'] = convergence_pass and smoothness_pass
        
        # Region classification
        if 'flat_vs_curved_regions' in self.test_results:
            regions = self.test_results['flat_vs_curved_regions']
            summary['regions_pass'] = regions['mean_classification_accuracy'] >= 0.6
        
        # Overall pass
        summary['overall_geometry_pass'] = all(summary.values()) if summary else False
        
        return summary
    
    def _print_geometry_summary(self, results: Dict[str, any]):
        """Print formatted geometry test summary"""
        print("\n" + "="*60)
        print("GEOMETRY LEARNING TEST SUMMARY")
        print("="*60)
        
        summary = results['pass_fail_summary']
        
        def status_emoji(passed):
            return "✅" if passed else "❌"
        
        # Christoffel matching
        if 'christoffel_matching' in results:
            r2 = results['christoffel_matching'].get('christoffel_r2', 0)
            print(f"{status_emoji(summary.get('christoffel_pass', False))} Christoffel Matching: R² = {r2:.3f}")
        
        # Curvature alignment  
        if 'curvature_alignment' in results:
            alignment = results['curvature_alignment']['mean_curvature_alignment']
            print(f"{status_emoji(summary.get('curvature_pass', False))} Curvature Alignment: {alignment:.3f}")
        
        # Path validation
        if 'path_validation' in results:
            convergence = results['path_validation']['mean_convergence_rate']
            print(f"{status_emoji(summary.get('path_pass', False))} Path Validation: Convergence = {convergence:.1%}")
        
        # Region classification
        if 'region_classification' in results:
            classification = results['region_classification']['mean_classification_accuracy']
            print(f"{status_emoji(summary.get('regions_pass', False))} Region Classification: {classification:.1%}")
        
        print(f"\n{status_emoji(summary.get('overall_geometry_pass', False))} OVERALL GEOMETRY: {'PASS' if summary.get('overall_geometry_pass', False) else 'FAIL'}")
        print("="*60)
    
    def create_geometry_visualization(self, save_path: str = "outputs/geometry_learning_results.png"):
        """
        Create comprehensive visualization of geometry learning results
        """
        if not self.test_results:
            print("❌ No test results available for visualization")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Geometry Learning Test Results', fontsize=16, fontweight='bold')
        
        # 1. Christoffel symbols comparison
        if 'christoffel_target_matching' in self.test_results and self.model.target_christoffel_computer:
            # Generate test points for visualization
            n_viz_points = 50
            c_test = torch.linspace(-1, 1, n_viz_points, device=self.device)
            wl_test = torch.zeros_like(c_test)  # Single wavelength for visualization
            
            with torch.no_grad():
                targets = self.model.target_christoffel_computer.get_target_christoffel(c_test, wl_test).cpu()
                actuals = self.model.christoffel_computer.interpolate(c_test, wl_test).cpu()
            
            axes[0, 0].plot(c_test.cpu(), targets, 'b-', label='Target', linewidth=2)
            axes[0, 0].plot(c_test.cpu(), actuals, 'r--', label='Learned', linewidth=2)
            axes[0, 0].set_xlabel('Concentration (normalized)')
            axes[0, 0].set_ylabel('Christoffel Symbol')
            axes[0, 0].set_title('Christoffel Symbol Matching')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Christoffel correlation plot
        if 'christoffel_target_matching' in self.test_results:
            # Scatter plot of target vs actual
            n_scatter = 100
            c_scatter = torch.rand(n_scatter, device=self.device) * 2 - 1
            wl_scatter = torch.rand(n_scatter, device=self.device) * 2 - 1
            
            if self.model.target_christoffel_computer:
                with torch.no_grad():
                    targets_scatter = self.model.target_christoffel_computer.get_target_christoffel(c_scatter, wl_scatter).cpu()
                    actuals_scatter = self.model.christoffel_computer.interpolate(c_scatter, wl_scatter).cpu()
                
                axes[0, 1].scatter(targets_scatter, actuals_scatter, alpha=0.6, s=20)
                min_val = min(targets_scatter.min(), actuals_scatter.min())
                max_val = max(targets_scatter.max(), actuals_scatter.max())
                axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                
                # Add correlation coefficient
                corr = self.test_results['christoffel_target_matching'].get('christoffel_correlation', 0)
                axes[0, 1].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[0, 1].transAxes,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            axes[0, 1].set_xlabel('Target Christoffel')
            axes[0, 1].set_ylabel('Learned Christoffel')
            axes[0, 1].set_title('Target vs Learned')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Sample geodesic trajectories
        if 'geodesic_path_validation' in self.test_results:
            # Show a few example geodesic paths
            c_sources = [-0.5, 0.0, 0.5]
            c_targets = [0.5, -0.5, -0.8]
            wl_norm = 0.0  # Middle wavelength
            
            for i, (c_src, c_tgt) in enumerate(zip(c_sources, c_targets)):
                with torch.no_grad():
                    c_src_tensor = torch.tensor([c_src], device=self.device, dtype=torch.float32)
                    c_tgt_tensor = torch.tensor([c_tgt], device=self.device, dtype=torch.float32)
                    wl_tensor = torch.tensor([wl_norm], device=self.device, dtype=torch.float32)
                    
                    result = self.model.forward(c_src_tensor, c_tgt_tensor, wl_tensor)
                    
                    if 'trajectories' in result:
                        trajectory = result['trajectories']  # [n_points, batch_size, 3]
                        c_path = trajectory[:, 0, 0].cpu().numpy()
                        A_path = trajectory[:, 0, 2].cpu().numpy()
                        
                        color = plt.cm.viridis(i / max(1, len(c_sources)-1))
                        axes[0, 2].plot(c_path, A_path, 'o-', color=color, alpha=0.8,
                                       label=f'{c_src:.1f} → {c_tgt:.1f}', markersize=3)
            
            axes[0, 2].set_xlabel('Concentration (normalized)')
            axes[0, 2].set_ylabel('Absorbance')  
            axes[0, 2].set_title('Sample Geodesic Paths')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Test scores summary
        test_categories = ['Christoffel', 'Curvature', 'Paths', 'Regions']
        test_scores = []
        test_colors = []
        
        if 'christoffel_target_matching' in self.test_results:
            r2 = self.test_results['christoffel_target_matching'].get('christoffel_r2', 0)
            score = max(0, min(100, r2 * 100))
            test_scores.append(score)
            test_colors.append('green' if score >= 70 else 'red')
        
        if 'metric_curvature_alignment' in self.test_results:
            alignment = self.test_results['metric_curvature_alignment']['mean_curvature_alignment']
            score = max(0, (alignment + 1) * 50)
            test_scores.append(score)
            test_colors.append('green' if score >= 65 else 'red')
        
        if 'geodesic_path_validation' in self.test_results:
            path = self.test_results['geodesic_path_validation']
            convergence = path['mean_convergence_rate']
            smoothness = path['mean_trajectory_smoothness']
            score = (convergence + smoothness) * 50
            test_scores.append(score)
            test_colors.append('green' if score >= 75 else 'red')
        
        if 'flat_vs_curved_regions' in self.test_results:
            classification = self.test_results['flat_vs_curved_regions']['mean_classification_accuracy']
            score = classification * 100
            test_scores.append(score)
            test_colors.append('green' if score >= 60 else 'red')
        
        if test_scores:
            bars = axes[1, 0].bar(test_categories[:len(test_scores)], test_scores, color=test_colors, alpha=0.7)
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].set_title('Geometry Test Scores')
            axes[1, 0].set_ylim(0, 100)
            
            # Add score labels on bars
            for bar, score in zip(bars, test_scores):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                               f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
            
            axes[1, 0].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(1, 3):
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Geometry learning visualization saved to {save_path}")
        
        # Show plot
        try:
            plt.show()
            subprocess.run(['open', save_path], check=False)
        except:
            pass
        
        return save_path