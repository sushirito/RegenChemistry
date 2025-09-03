"""
Geodesic Neural ODE for M1 Mac
Optimized implementation for Apple Silicon with MPS acceleration
"""

__version__ = "1.0.0"
__author__ = "Geodesic NODE Team"
__description__ = "M1 Mac optimized Geodesic-Coupled Spectral NODE system"

from .core.christoffel_computer import ChristoffelComputer
from .core.geodesic_integrator import GeodesicIntegrator
from .core.shooting_solver import ShootingSolver
from .models.metric_network import MetricNetwork
from .models.spectral_flow_network import SpectralFlowNetwork
from .models.geodesic_model import GeodesicNODE

__all__ = [
    'ChristoffelComputer',
    'GeodesicIntegrator', 
    'ShootingSolver',
    'MetricNetwork',
    'SpectralFlowNetwork',
    'GeodesicNODE'
]