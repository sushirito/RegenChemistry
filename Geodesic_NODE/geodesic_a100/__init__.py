"""
Geodesic NODE A100-Optimized Implementation
Ultra-parallel MVP for NVIDIA A100 GPU
"""

__version__ = "1.0.0"
__author__ = "Geodesic NODE Team"

from .core.device_manager import DeviceManager
from .models.geodesic_model import GeodesicNODE

__all__ = [
    "DeviceManager",
    "GeodesicNODE"
]