#!/usr/bin/env python3
"""
Environment validation and setup for M1 Mac
Checks PyTorch MPS availability and system compatibility
"""

import torch
import platform
import sys
from pathlib import Path


def check_m1_compatibility():
    """Check if system is compatible with M1 optimizations"""
    print("🔍 M1 Mac Compatibility Check")
    print("=" * 40)
    
    # System information
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    
    # Check for Apple Silicon
    is_apple_silicon = platform.machine() == 'arm64'
    print(f"Apple Silicon: {'✅ Yes' if is_apple_silicon else '❌ No'}")
    
    # Check MPS availability
    mps_available = torch.backends.mps.is_available()
    print(f"MPS Available: {'✅ Yes' if mps_available else '❌ No'}")
    
    if mps_available:
        # Test MPS functionality
        try:
            x = torch.randn(100, 100, device='mps')
            y = torch.randn(100, 100, device='mps')
            z = torch.matmul(x, y)
            print("MPS Test: ✅ Working")
        except Exception as e:
            print(f"MPS Test: ❌ Failed - {e}")
            mps_available = False
    
    # Check memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"Total Memory: {memory_gb:.1f} GB")
        
        if memory_gb >= 16:
            config_rec = "performance"
        elif memory_gb >= 8:
            config_rec = "memory_optimized" 
        else:
            config_rec = "memory_optimized (with smaller batch sizes)"
            
        print(f"Recommended Config: {config_rec}")
        
    except ImportError:
        print("Memory Info: ❌ psutil not available")
    
    # Overall recommendation
    print("\n🎯 Recommendation:")
    if is_apple_silicon and mps_available:
        print("✅ System fully compatible with M1 optimizations")
        print("   Run: python main.py --config performance")
    elif is_apple_silicon:
        print("⚠️  Apple Silicon detected but MPS unavailable")
        print("   Update PyTorch: pip install torch --upgrade")
    else:
        print("❌ Not Apple Silicon - use CPU or CUDA version")
        

if __name__ == "__main__":
    check_m1_compatibility()