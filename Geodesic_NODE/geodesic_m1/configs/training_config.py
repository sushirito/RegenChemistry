"""Training loop parameters for M1 Mac geodesic NODE"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class TrainingConfig:
    """Training configuration for M1 Mac"""
    
    # Training loop parameters
    epochs: int = 100
    batch_size: int = 1024
    validation_batch_size: int = 512
    
    # Optimization parameters
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    
    # Scheduler parameters
    use_scheduler: bool = True
    scheduler_type: str = 'cosine'  # 'cosine', 'step', 'plateau'
    scheduler_patience: int = 10    # for plateau scheduler
    
    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-6
    
    # Checkpointing
    save_best_only: bool = True
    checkpoint_interval: int = 10
    
    # Validation
    validation_interval: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.__dict__.copy()