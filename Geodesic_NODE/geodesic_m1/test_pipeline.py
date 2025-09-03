#!/usr/bin/env python3
"""
Test script to verify the complete pipeline setup
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        # Core modules
        from geodesic_m1.configs.m1_config import create_m1_config, get_memory_optimized_config
        print("✓ Configs imported")
        
        from geodesic_m1.training.trainer import M1Trainer
        print("✓ Trainer imported")
        
        from geodesic_m1.models.geodesic_model import GeodesicNODE
        print("✓ Model imported")
        
        # New modules
        from geodesic_m1.utils.metrics import compute_all_metrics
        print("✓ Metrics imported")
        
        from geodesic_m1.validation.evaluator import run_complete_validation
        print("✓ Validation imported")
        
        from geodesic_m1.visualization.training_plots import plot_training_history
        print("✓ Training plots imported")
        
        from geodesic_m1.visualization.comparison_3d import create_3d_comparison_plot
        print("✓ 3D visualization imported")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        return False


def test_directories():
    """Test that output directories exist"""
    print("\nTesting directories...")
    
    required_dirs = [
        'outputs',
        'outputs/models',
        'outputs/training_logs',
        'outputs/metrics',
        'outputs/visualizations',
        'outputs/predictions',
        'checkpoints'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path} exists")
        else:
            print(f"✗ {dir_path} missing")
            all_exist = False
    
    if all_exist:
        print("\n✅ All directories exist!")
    else:
        print("\n⚠️  Some directories missing - they will be created during training")
    
    return all_exist


def test_config():
    """Test configuration settings"""
    print("\nTesting configuration...")
    
    from geodesic_m1.configs.m1_config import get_memory_optimized_config
    
    config = get_memory_optimized_config()
    
    print(f"✓ Device: {config.device}")
    print(f"✓ Epochs: {config.epochs}")
    print(f"✓ Batch size: {config.recommended_batch_size}")
    print(f"✓ Grid size: {config.christoffel_grid_size}")
    
    if config.epochs == 25:
        print("\n✅ Configuration correct (25 epochs)")
    else:
        print(f"\n⚠️  Epochs set to {config.epochs} (expected 25)")
    
    return True


def main():
    """Run all tests"""
    print("="*50)
    print("GEODESIC NODE M1 PIPELINE TEST")
    print("="*50)
    
    success = True
    
    # Run tests
    success &= test_imports()
    success &= test_directories()
    success &= test_config()
    
    print("\n" + "="*50)
    if success:
        print("✅ PIPELINE READY!")
        print("\nTo train the full pipeline, run:")
        print("  python geodesic_m1/main.py --config memory_optimized --epochs 25")
    else:
        print("⚠️  Some issues detected - please fix before running")
    print("="*50)


if __name__ == "__main__":
    main()