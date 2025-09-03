"""A100-specific configuration parameters"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class A100Config:
    """A100-specific hardware and optimization configuration"""
    
    # Device settings
    device: str = "cuda"
    mixed_precision: bool = True
    
    # Memory management
    memory_fraction: float = 0.95
    gradient_checkpointing: bool = False
    
    # Batch processing
    mega_batch_size: int = 18030  # All geodesics at once
    micro_batch_size: int = 2048  # For gradient accumulation
    
    # Parallelization
    num_workers: int = 0  # Keep data on GPU
    pin_memory: bool = False  # Already on GPU
    
    # Tensor Core optimization
    use_tensor_cores: bool = True
    tensor_core_precision: str = "fp16"  # fp16 or bf16
    
    # Performance settings
    benchmark_cudnn: bool = True
    deterministic: bool = False  # Disable for speed
    
    # Multi-stream execution
    num_streams: int = 4
    
    # Grid pre-computation
    christoffel_grid_size: tuple = (2000, 601)
    christoffel_precision: str = "fp16"  # Store in half precision
    
    # Memory pool settings
    enable_memory_pool: bool = True
    pool_size_gb: float = 10.0
    
    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.mega_batch_size % self.micro_batch_size == 0, \
            "mega_batch_size must be divisible by micro_batch_size"
        assert self.memory_fraction > 0 and self.memory_fraction <= 1, \
            "memory_fraction must be between 0 and 1"
        assert self.tensor_core_precision in ["fp16", "bf16"], \
            "tensor_core_precision must be fp16 or bf16"