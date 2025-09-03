"""
Core mathematical components for geodesic computation on M1 Mac
"""

from .christoffel_computer import ChristoffelComputer
from .geodesic_integrator import GeodesicIntegrator
from .shooting_solver import ShootingSolver
from .device_manager import M1DeviceManager

__all__ = [
    'ChristoffelComputer',
    'GeodesicIntegrator',
    'ShootingSolver',
    'M1DeviceManager'
]