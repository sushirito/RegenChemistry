#!/usr/bin/env python3
"""
Simple test to verify imports and basic functionality
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add geodesic_m1 to path
sys.path.append(str(Path(__file__).parent / 'geodesic_m1'))

try:
    print("Testing imports...")
    from geodesic_m1.core.target_christoffel import TargetChristoffelComputer
    print("✅ TargetChristoffelComputer imported")
    
    from geodesic_m1.models.geodesic_model import GeodesicNODE  
    print("✅ GeodesicNODE imported")
    
    from geodesic_m1.tests.test_data_adherence import DataAdherenceTests
    print("✅ DataAdherenceTests imported")
    
    from geodesic_m1.tests.test_geometry_learning import GeometryLearningTests
    print("✅ GeometryLearningTests imported")
    
    print("\nTesting basic functionality...")
    
    # Create simple synthetic data
    concentrations = np.array([0.0, 10.0, 20.0])
    wavelengths = np.array([400.0, 500.0, 600.0])
    absorbance_data = np.random.rand(3, 3) * 0.1 + 0.01
    
    print("✅ Created synthetic data")
    
    # Test target Christoffel computation
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    target_computer = TargetChristoffelComputer(
        concentrations=concentrations,
        wavelengths=wavelengths,
        absorbance_data=absorbance_data,
        device=device
    )
    print("✅ TargetChristoffelComputer created")
    
    # Test target computation
    c_test = torch.tensor([0.0], device=device)
    wl_test = torch.tensor([0.0], device=device)
    targets = target_computer.get_target_christoffel(c_test, wl_test)
    print(f"✅ Target computation works: {targets}")
    
    # Test model creation
    model = GeodesicNODE(
        metric_hidden_dims=[8, 16],  # Very small
        flow_hidden_dims=[8, 16],
        n_trajectory_points=5,
        shooting_max_iter=2,
        device=device,
        concentrations=concentrations,
        wavelengths=wavelengths,
        absorbance_matrix=absorbance_data
    )
    print("✅ GeodesicNODE created successfully")
    
    # Check that target computer was integrated
    if model.target_christoffel_computer is not None:
        print("✅ Target Christoffel computer integrated into model")
    else:
        print("❌ Target Christoffel computer not integrated")
    
    print("\n🎉 Basic functionality test PASSED!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)