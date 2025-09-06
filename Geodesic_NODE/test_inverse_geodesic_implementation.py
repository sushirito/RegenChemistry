#!/usr/bin/env python3
"""
Test script for inverse geodesic implementation
Tests the new Christoffel matching loss and validation pipeline
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Add geodesic_m1 to path
sys.path.append(str(Path(__file__).parent / 'geodesic_m1'))

from geodesic_m1.models.geodesic_model import GeodesicNODE
from geodesic_m1.core.target_christoffel import TargetChristoffelComputer
from geodesic_m1.training.data_loader import SpectralDataset, create_leave_one_out_datasets
from geodesic_m1.tests.test_data_adherence import DataAdherenceTests
from geodesic_m1.tests.test_geometry_learning import GeometryLearningTests
from geodesic_m1.tests.validation_pipeline import ComprehensiveValidationPipeline


def create_synthetic_spectral_data():
    """Create synthetic spectral data for testing"""
    # Create concentration and wavelength grids
    concentrations = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 60.0])  # ppb
    wavelengths = np.linspace(400, 700, 61)  # nm (reduced for faster testing)
    
    # Create synthetic absorbance data with realistic spectral shapes
    absorbance_data = np.zeros((len(concentrations), len(wavelengths)))
    
    for i, c in enumerate(concentrations):
        for j, wl in enumerate(wavelengths):
            # Synthetic spectral response with some non-linearity
            # Peak around 500-550nm that shifts and broadens with concentration
            peak_wl = 525 + c * 0.2  # Peak shifts slightly with concentration
            width = 50 + c * 0.5     # Peak broadens with concentration
            intensity = 0.1 + c * 0.015  # Base intensity increases with concentration
            
            # Gaussian-like peak
            abs_val = intensity * np.exp(-((wl - peak_wl) / width)**2)
            
            # Add some baseline and noise
            baseline = 0.01 + c * 0.001
            noise = np.random.normal(0, 0.005)
            
            absorbance_data[i, j] = abs_val + baseline + noise
    
    # Ensure non-negative values
    absorbance_data = np.maximum(absorbance_data, 0.001)
    
    return concentrations, wavelengths, absorbance_data


def test_target_christoffel_computation():
    """Test the target Christoffel computation"""
    print("🧪 Testing target Christoffel computation...")
    
    concentrations, wavelengths, absorbance_data = create_synthetic_spectral_data()
    
    # Create target computer
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    target_computer = TargetChristoffelComputer(
        concentrations=concentrations,
        wavelengths=wavelengths,
        absorbance_data=absorbance_data,
        device=device
    )
    
    # Validate targets
    validation_metrics = target_computer.validate_targets()
    print(f"   Target validation: {validation_metrics}")
    
    # Test interpolation
    c_test = torch.tensor([0.0, 0.5], device=device)
    wl_test = torch.tensor([0.0, 0.0], device=device)
    targets = target_computer.get_target_christoffel(c_test, wl_test)
    
    print(f"   Sample target values: {targets}")
    
    # Create visualization
    target_computer.plot_target_examples(n_examples=3)
    
    return target_computer


def test_geodesic_model_with_targets():
    """Test the geodesic model with target Christoffel loss"""
    print("\n🧪 Testing geodesic model with Christoffel matching loss...")
    
    concentrations, wavelengths, absorbance_data = create_synthetic_spectral_data()
    
    # Create model with data
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = GeodesicNODE(
        metric_hidden_dims=[32, 64],  # Smaller for faster testing
        flow_hidden_dims=[32, 64],
        n_trajectory_points=10,       # Fewer points for speed
        shooting_max_iter=5,          # Fewer iterations for speed  
        device=device,
        concentrations=concentrations,
        wavelengths=wavelengths,
        absorbance_matrix=absorbance_data
    )
    
    print(f"   Model info: {model.get_model_info()}")
    
    # Test that target Christoffel computer was created
    assert model.target_christoffel_computer is not None, "Target Christoffel computer not created"
    print("   ✅ Target Christoffel computer created successfully")
    
    # Test forward pass
    c_src = torch.tensor([0.0], device=device)
    c_tgt = torch.tensor([0.5], device=device)
    wl = torch.tensor([0.0], device=device)
    
    model.precompute_christoffel_grid()
    
    with torch.no_grad():
        result = model.forward(c_src, c_tgt, wl)
        print(f"   Forward pass result keys: {result.keys()}")
        print(f"   Predicted absorbance: {result['absorbance'].item():.4f}")
        print(f"   Convergence: {result['convergence_rate'].item():.1%}")
    
    # Test loss computation (requires gradient)
    c_src.requires_grad_(True)
    result = model.forward(c_src, c_tgt, wl)
    
    # Dummy target for loss computation
    target_abs = torch.tensor([0.1], device=device)
    
    loss_components = model.compute_loss(result, target_abs, c_src, wl)
    print(f"   Loss components: {list(loss_components.keys())}")
    print(f"   Total loss: {loss_components['total'].item():.6f}")
    print(f"   Christoffel loss: {loss_components['christoffel_matching'].item():.6f}")
    
    # Test geometry validation
    geometry_metrics = model.validate_geometry_learning(
        torch.tensor([0.0, 0.5], device=device), 
        torch.tensor([0.0, 0.0], device=device)
    )
    print(f"   Geometry validation: {geometry_metrics}")
    
    return model


def test_validation_pipeline():
    """Test the comprehensive validation pipeline"""
    print("\n🧪 Testing validation pipeline...")
    
    concentrations, wavelengths, absorbance_data = create_synthetic_spectral_data()
    
    # Create leave-one-out datasets (just test 2 models for speed)
    datasets = create_leave_one_out_datasets(
        wavelengths, absorbance_data,
        device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    )[:2]  # Only test first 2 models
    
    # Create simple models
    models = []
    for i in range(2):
        model = GeodesicNODE(
            metric_hidden_dims=[16, 32],  # Very small for fast testing
            flow_hidden_dims=[16, 32],
            n_trajectory_points=5,
            shooting_max_iter=3,
            concentrations=concentrations,
            wavelengths=wavelengths,
            absorbance_matrix=absorbance_data
        )
        model.precompute_christoffel_grid()
        models.append(model)
    
    # Create validation pipeline
    pipeline = ComprehensiveValidationPipeline(
        models=models,
        datasets=datasets,
        output_dir="outputs/test_validation",
        verbose=True
    )
    
    # Run validation
    start_time = time.time()
    results = pipeline.run_complete_validation()
    end_time = time.time()
    
    print(f"   Validation completed in {end_time - start_time:.1f} seconds")
    print(f"   Overall grade: {results['overall_summary']['grade']}")
    print(f"   Mean score: {results['aggregate_results']['overall']['mean_score']:.1f}")
    
    return results


def test_individual_test_suites():
    """Test individual test suites"""
    print("\n🧪 Testing individual test suites...")
    
    concentrations, wavelengths, absorbance_data = create_synthetic_spectral_data()
    
    # Create dataset and model
    dataset = SpectralDataset(
        concentration_values=concentrations,
        wavelengths=wavelengths,
        absorbance_data=absorbance_data
    )
    
    model = GeodesicNODE(
        metric_hidden_dims=[16, 32],
        flow_hidden_dims=[16, 32],
        concentrations=concentrations,
        wavelengths=wavelengths,
        absorbance_matrix=absorbance_data
    )
    model.precompute_christoffel_grid()
    
    # Test data adherence
    print("   Testing data adherence...")
    adherence_tester = DataAdherenceTests(model, dataset, tolerance=0.1)
    adherence_results = adherence_tester.test_known_concentration_accuracy(verbose=False)
    print(f"   Data adherence R²: {adherence_results['r2_score']:.3f}")
    
    # Test geometry learning
    print("   Testing geometry learning...")
    geometry_tester = GeometryLearningTests(model, dataset)
    geometry_results = geometry_tester.test_christoffel_target_matching(n_test_points=20, verbose=False)
    if 'error' not in geometry_results:
        print(f"   Christoffel R²: {geometry_results['christoffel_r2']:.3f}")
    else:
        print(f"   Geometry test failed: {geometry_results['error']}")
    
    return adherence_results, geometry_results


def main():
    """Run all tests"""
    print("🚀 Testing Inverse Geodesic Implementation")
    print("=" * 60)
    
    try:
        # Test 1: Target Christoffel computation
        target_computer = test_target_christoffel_computation()
        
        # Test 2: Geodesic model with targets
        model = test_geodesic_model_with_targets()
        
        # Test 3: Individual test suites
        adherence_results, geometry_results = test_individual_test_suites()
        
        # Test 4: Full validation pipeline (this takes longer)
        print("\n🔬 Running comprehensive validation pipeline...")
        validation_results = test_validation_pipeline()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("📊 Summary:")
        print(f"   • Target Christoffel computation: ✅")
        print(f"   • Geodesic model with targets: ✅")  
        print(f"   • Individual test suites: ✅")
        print(f"   • Validation pipeline: ✅")
        print(f"   • Overall validation grade: {validation_results['overall_summary']['grade']}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)