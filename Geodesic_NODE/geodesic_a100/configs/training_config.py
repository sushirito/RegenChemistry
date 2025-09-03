"""Training configuration parameters"""

from dataclasses import dataclass
from typing import Optional


@dataclass 
class TrainingConfig:
    """Training loop configuration"""
    
    # Training parameters
    num_epochs: int = 500
    learning_rate_metric: float = 5e-4
    learning_rate_decoder: float = 1e-3
    weight_decay: float = 1e-5
    
    # Gradient settings
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 9  # 18030 / 2048 ≈ 9
    
    # Mixed precision
    use_amp: bool = True
    loss_scale_init: float = 2**16
    loss_scale_growth_factor: float = 2.0
    loss_scale_backoff_factor: float = 0.5
    loss_scale_growth_interval: int = 100
    
    # Optimizer settings
    optimizer: str = "adam"
    adam_betas: tuple = (0.9, 0.999)
    adam_eps: float = 1e-8
    
    # Scheduler settings
    scheduler: str = "cosine"
    warmup_epochs: int = 10
    min_lr: float = 1e-6
    
    # Loss weights
    reconstruction_weight: float = 1.0
    smoothness_weight: float = 0.01
    path_length_weight: float = 0.001
    
    # Validation
    validation_frequency: int = 10  # Validate every N epochs
    save_frequency: int = 50  # Save checkpoint every N epochs
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 50
    min_delta: float = 1e-6
    
    # Logging
    log_frequency: int = 10  # Log every N batches
    use_tensorboard: bool = False
    log_dir: str = "./logs"
    
    # Data settings
    num_concentrations: int = 6
    num_wavelengths: int = 601
    concentration_values: list = None  # Will set in __post_init__
    
    def __post_init__(self):
        """Set default concentration values"""
        if self.concentration_values is None:
            self.concentration_values = [0, 10, 20, 30, 40, 60]  # ppb
            
    def validate(self) -> None:
        """Validate training configuration"""
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert self.learning_rate_metric > 0, "learning_rate_metric must be positive"
        assert self.learning_rate_decoder > 0, "learning_rate_decoder must be positive"
        assert self.optimizer in ["adam", "adamw", "sgd"], f"Unknown optimizer: {self.optimizer}"
        assert self.scheduler in ["cosine", "linear", "constant", "exponential"], \
            f"Unknown scheduler: {self.scheduler}"
        assert len(self.concentration_values) == self.num_concentrations, \
            "concentration_values length must match num_concentrations"