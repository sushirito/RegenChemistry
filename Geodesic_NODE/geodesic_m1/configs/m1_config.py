"""
M1 Mac specific configuration and hyperparameters
Optimized for Apple Silicon MPS acceleration
"""

import torch
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class M1Config:
    """M1 Mac optimized configuration for geodesic NODE"""
    
    # Device configuration
    # Temporarily use CPU due to MPS/torchdiffeq compatibility issues
    device: torch.device = field(default_factory=lambda: torch.device('cpu'))
    use_mixed_precision: bool = True
    
    # Memory configuration
    max_cache_memory_mb: float = 8192  # 8GB for caching
    recommended_batch_size: int = 1024  # Optimal for M1
    gradient_accumulation_steps: int = 2
    
    # Performance optimization
    num_workers: int = 0  # MPS works better with 0 workers
    pin_memory: bool = False  # Not needed for unified memory
    persistent_workers: bool = False
    
    # Christoffel grid configuration (same as A100 for mathematical equivalence)
    christoffel_grid_size: tuple = (2000, 601)
    use_half_precision_grid: bool = True
    
    # Training configuration
    epochs: int = 25  # Reduced for faster testing
    early_stopping_patience: int = 20
    checkpoint_interval: int = 10  # Save every 10 epochs
    
    # Learning rates (optimized for M1)
    metric_lr: float = 3e-4  # Slightly lower than A100
    flow_lr: float = 8e-4    # Slightly lower than A100
    weight_decay: float = 1e-5
    
    # Loss weights (same as A100)
    smoothness_weight: float = 0.01
    bounds_weight: float = 0.001
    path_length_weight: float = 0.001
    
    # Shooting method parameters (same as A100)
    shooting_max_iter: int = 10
    shooting_tolerance: float = 1e-4
    shooting_learning_rate: float = 0.5
    
    # ODE integration parameters
    n_trajectory_points: int = 50
    ode_rtol: float = 1e-5
    ode_atol: float = 1e-7
    use_adjoint: bool = True
    
    # Data augmentation
    noise_level: float = 0.01
    wavelength_shift: float = 0.02  # Small wavelength perturbations
    concentration_noise: float = 0.005  # Small concentration perturbations
    
    # Validation configuration
    validation_batch_size: int = 512
    run_validation_every: int = 5  # epochs
    
    # Logging and monitoring
    log_interval: int = 10
    profile_training: bool = True
    monitor_memory: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, torch.device):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'M1Config':
        """Create config from dictionary"""
        # Handle device string conversion
        if 'device' in config_dict and isinstance(config_dict['device'], str):
            config_dict['device'] = torch.device(config_dict['device'])
            
        return cls(**config_dict)
    
    def get_optimizer_config(self) -> Dict[str, Dict[str, Any]]:
        """Get optimizer configuration"""
        return {
            'metric': {
                'lr': self.metric_lr,
                'weight_decay': self.weight_decay,
                'optimizer_type': 'adam'
            },
            'flow': {
                'lr': self.flow_lr,
                'weight_decay': self.weight_decay,
                'optimizer_type': 'adam'
            }
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            'metric_hidden_dims': [128, 256],  # M1-optimized dimensions
            'flow_hidden_dims': [64, 128],
            'n_trajectory_points': self.n_trajectory_points,
            'shooting_max_iter': self.shooting_max_iter,
            'shooting_tolerance': self.shooting_tolerance,
            'shooting_learning_rate': self.shooting_learning_rate,
            'christoffel_grid_size': self.christoffel_grid_size,
            'device': self.device,
            'use_adjoint': self.use_adjoint
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return {
            'epochs': self.epochs,
            'batch_size': self.recommended_batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'early_stopping_patience': self.early_stopping_patience,
            'checkpoint_interval': self.checkpoint_interval,
            'validation_batch_size': self.validation_batch_size,
            'run_validation_every': self.run_validation_every,
            'log_interval': self.log_interval,
            'use_mixed_precision': self.use_mixed_precision
        }
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data loading configuration"""
        return {
            'batch_size': self.recommended_batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'persistent_workers': self.persistent_workers,
            'noise_level': self.noise_level,
            'wavelength_shift': self.wavelength_shift,
            'concentration_noise': self.concentration_noise
        }
    
    def validate_config(self) -> bool:
        """Validate configuration consistency"""
        try:
            # Check device availability
            if self.device.type == 'mps' and not torch.backends.mps.is_available():
                print("⚠️  MPS not available, falling back to CPU")
                self.device = torch.device('cpu')
                self.use_mixed_precision = False
                
            # Adjust batch size based on available memory
            if self.recommended_batch_size > 4096:
                print("⚠️  Batch size too large for M1, reducing to 2048")
                self.recommended_batch_size = 2048
                
            # Validate grid size
            grid_memory_mb = (self.christoffel_grid_size[0] * 
                            self.christoffel_grid_size[1] * 
                            (2 if self.use_half_precision_grid else 4)) / (1024**2)
            
            if grid_memory_mb > 50:  # 50MB limit for grid
                print(f"⚠️  Christoffel grid may be too large: {grid_memory_mb:.1f} MB")
                
            return True
            
        except Exception as e:
            print(f"❌ Config validation failed: {e}")
            return False
    
    def print_config(self):
        """Print configuration summary"""
        print("🔧 M1 Configuration:")
        print(f"   Device: {self.device}")
        print(f"   Mixed Precision: {self.use_mixed_precision}")
        print(f"   Batch Size: {self.recommended_batch_size}")
        print(f"   Christoffel Grid: {self.christoffel_grid_size}")
        print(f"   Training Epochs: {self.epochs}")
        print(f"   Learning Rates: Metric={self.metric_lr}, Flow={self.flow_lr}")
        print(f"   Memory Budget: {self.max_cache_memory_mb} MB")


# Default configuration instance
default_m1_config = M1Config()


def create_m1_config(custom_params: Optional[Dict[str, Any]] = None) -> M1Config:
    """
    Create M1 configuration with optional custom parameters
    
    Args:
        custom_params: Optional dictionary of custom parameters
        
    Returns:
        Configured M1Config instance
    """
    if custom_params is None:
        config = M1Config()
    else:
        # Start with defaults and update with custom params
        config_dict = default_m1_config.to_dict()
        config_dict.update(custom_params)
        config = M1Config.from_dict(config_dict)
        
    # Validate and adjust configuration
    config.validate_config()
    
    return config


def get_memory_optimized_config() -> M1Config:
    """Get memory-optimized configuration for lower memory M1 Macs"""
    return create_m1_config({
        'recommended_batch_size': 256,
        'max_cache_memory_mb': 2048,  # 2GB
        'christoffel_grid_size': (500, 601),  # Reduced concentration, full wavelength resolution
        'gradient_accumulation_steps': 8,  # Compensate for smaller batch
        'epochs': 25  # Reduced for faster testing
    })


def get_performance_optimized_config() -> M1Config:
    """Get performance-optimized configuration for higher memory M1 Macs"""
    return create_m1_config({
        'recommended_batch_size': 2048,
        'max_cache_memory_mb': 12288,  # 12GB
        'christoffel_grid_size': (2000, 601),  # Full grid
        'gradient_accumulation_steps': 1,
        'epochs': 75  # Fewer epochs with larger batches
    })