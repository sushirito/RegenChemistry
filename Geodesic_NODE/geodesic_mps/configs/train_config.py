#!/usr/bin/env python3
"""
Training Configuration for MPS-Optimized Geodesic NODE
"""

from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    metric_hidden_dim: int = 8
    metric_n_layers: int = 2
    spectral_hidden_dim: int = 16
    spectral_n_layers: int = 1
    n_trajectory_points: int = 11
    shooting_max_iter: int = 10
    use_christoffel_cache: bool = False


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # Batch sizes
    mega_batch_size: int = 2048
    micro_batch_size: int = 256
    gradient_accumulation_steps: int = 8
    
    # Optimization
    metric_lr: float = 5e-4
    spectral_lr: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    
    # Training schedule
    n_epochs: int = 100
    warmup_epochs: int = 5
    
    # Data sampling
    wavelength_subset: Optional[int] = None  # None = use all 601
    data_sampling_ratio: float = 0.1  # Use 10% of data per epoch
    
    # Loss weights
    reconstruction_weight: float = 1.0
    smoothness_weight: float = 0.01
    bounds_weight: float = 0.001
    path_weight: float = 0.001
    
    # Checkpointing
    checkpoint_interval: int = 2
    validation_interval: int = 2
    
    # Early stopping
    patience: int = 20
    min_delta: float = 1e-6


@dataclass
class MPSConfig:
    """MPS device-specific configuration"""
    device: str = "mps"
    use_mixed_precision: bool = False  # MPS doesn't fully support yet
    compile_model: bool = True
    cache_christoffel: bool = True
    parallel_wavelengths: int = 601
    concurrent_transitions: int = 30


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    model: ModelConfig = None
    training: TrainingConfig = None
    mps: MPSConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.mps is None:
            self.mps = MPSConfig()
    
    # Paths
    data_path: str = "data/0.30MB_AuNP_As.csv"
    checkpoint_dir: str = "checkpoints/"
    log_dir: str = "logs/"
    
    # Experiment info
    experiment_name: str = "geodesic_mps"
    seed: int = 42
    
    def get_device(self) -> torch.device:
        """Get configured device"""
        if self.mps.device == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        elif self.mps.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def get_effective_batch_size(self) -> int:
        """Calculate effective batch size with gradient accumulation"""
        return self.training.micro_batch_size * self.training.gradient_accumulation_steps
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging"""
        return {
            'model': {
                'metric_hidden_dim': self.model.metric_hidden_dim,
                'metric_n_layers': self.model.metric_n_layers,
                'spectral_hidden_dim': self.model.spectral_hidden_dim,
                'spectral_n_layers': self.model.spectral_n_layers,
                'n_trajectory_points': self.model.n_trajectory_points,
                'shooting_max_iter': self.model.shooting_max_iter,
            },
            'training': {
                'mega_batch_size': self.training.mega_batch_size,
                'micro_batch_size': self.training.micro_batch_size,
                'effective_batch_size': self.get_effective_batch_size(),
                'metric_lr': self.training.metric_lr,
                'spectral_lr': self.training.spectral_lr,
                'n_epochs': self.training.n_epochs,
            },
            'device': str(self.get_device()),
        }


def get_quick_test_config() -> ExperimentConfig:
    """Get configuration for quick testing"""
    config = ExperimentConfig()
    config.training.n_epochs = 10
    config.training.mega_batch_size = 128
    config.training.micro_batch_size = 32
    config.training.gradient_accumulation_steps = 4
    config.training.data_sampling_ratio = 0.1  # Use 10% of data
    config.model.n_trajectory_points = 5
    config.model.shooting_max_iter = 3
    return config


def get_full_training_config() -> ExperimentConfig:
    """Get configuration for full training"""
    config = ExperimentConfig()
    config.training.n_epochs = 500
    config.training.data_sampling_ratio = 1.0
    config.model.use_christoffel_cache = True
    return config


def get_benchmark_config() -> ExperimentConfig:
    """Get configuration for benchmarking"""
    config = ExperimentConfig()
    config.training.n_epochs = 1
    config.training.mega_batch_size = 4096
    config.training.micro_batch_size = 512
    config.model.n_trajectory_points = 3
    config.model.shooting_max_iter = 2
    return config


if __name__ == "__main__":
    # Test configuration
    print("Testing Training Configuration...")
    
    # Default config
    config = ExperimentConfig()
    print(f"\nDefault configuration:")
    print(f"Device: {config.get_device()}")
    print(f"Effective batch size: {config.get_effective_batch_size()}")
    
    # Quick test config
    quick_config = get_quick_test_config()
    print(f"\nQuick test configuration:")
    print(f"Epochs: {quick_config.training.n_epochs}")
    print(f"Batch size: {quick_config.training.micro_batch_size}")
    
    # Full config
    full_config = get_full_training_config()
    print(f"\nFull training configuration:")
    print(f"Epochs: {full_config.training.n_epochs}")
    print(f"Data ratio: {full_config.training.data_sampling_ratio}")
    
    # Export to dict
    config_dict = config.to_dict()
    print(f"\nConfiguration dictionary:")
    for key, value in config_dict.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")