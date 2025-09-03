"""
Device Manager for A100-specific optimizations
Handles GPU configuration, mixed precision, and memory management
"""

import torch
import warnings
from typing import Optional, Dict, Any


class DeviceManager:
    """Manages A100-specific device configuration and optimization"""
    
    def __init__(self, force_cpu: bool = False):
        """
        Initialize device manager with A100 optimizations
        
        Args:
            force_cpu: Force CPU usage even if GPU available
        """
        self.device = self._setup_device(force_cpu)
        self.is_cuda = self.device.type == 'cuda'
        self.device_properties = self._get_device_properties()
        
        # Configure optimizations
        if self.is_cuda:
            self._configure_cuda_optimizations()
            
    def _setup_device(self, force_cpu: bool) -> torch.device:
        """Setup and validate device"""
        if force_cpu:
            print("Forcing CPU usage")
            return torch.device('cpu')
            
        if not torch.cuda.is_available():
            warnings.warn("CUDA not available, falling back to CPU")
            return torch.device('cpu')
            
        # Get device properties
        device_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {device_name}")
        
        # Check if it's an A100
        if 'A100' in device_name:
            print("A100 detected - enabling optimizations")
        else:
            warnings.warn(f"Non-A100 GPU detected ({device_name}), performance may vary")
            
        return torch.device('cuda:0')
        
    def _get_device_properties(self) -> Dict[str, Any]:
        """Get device properties for optimization"""
        if not self.is_cuda:
            return {}
            
        props = torch.cuda.get_device_properties(0)
        return {
            'name': props.name,
            'total_memory_gb': props.total_memory / (1024**3),
            'multiprocessor_count': props.multi_processor_count,
            'cuda_cores': props.multi_processor_count * 64,  # Approximate for A100
            'memory_bandwidth_gb': 1555 if 'A100' in props.name else 900,  # A100 specific
            'supports_fp16': True,
            'supports_bf16': torch.cuda.is_bf16_supported(),
            'tensor_cores': 432 if 'A100' in props.name else 0
        }
        
    def _configure_cuda_optimizations(self):
        """Configure CUDA-specific optimizations for A100"""
        # Enable TF32 for Ampere GPUs (A100)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cudNN autotuner for optimal convolution algorithms
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        # Set memory fraction to avoid OOM
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        print("A100 optimizations enabled:")
        print(f"  - TF32: Enabled")
        print(f"  - cuDNN benchmark: Enabled") 
        print(f"  - Memory fraction: 95%")
        print(f"  - Available memory: {self.get_available_memory():.1f} GB")
        
    def get_available_memory(self) -> float:
        """Get available GPU memory in GB"""
        if not self.is_cuda:
            return 0.0
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        if not self.is_cuda:
            return {}
            
        return {
            'allocated_gb': torch.cuda.memory_allocated() / (1024**3),
            'reserved_gb': torch.cuda.memory_reserved() / (1024**3),
            'free_gb': (torch.cuda.get_device_properties(0).total_memory - 
                       torch.cuda.memory_allocated()) / (1024**3)
        }
        
    def clear_cache(self):
        """Clear GPU cache"""
        if self.is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
    def get_optimal_batch_size(self, model: torch.nn.Module, 
                              sample_input_shape: tuple,
                              max_batch_size: int = 8192) -> int:
        """
        Find optimal batch size for given model
        
        Args:
            model: Model to test
            sample_input_shape: Shape of single input sample
            max_batch_size: Maximum batch size to try
            
        Returns:
            Optimal batch size that fits in memory
        """
        if not self.is_cuda:
            return 32  # Conservative CPU batch size
            
        model = model.to(self.device)
        model.eval()
        
        # Binary search for optimal batch size
        low, high = 1, max_batch_size
        optimal = 1
        
        with torch.no_grad():
            while low <= high:
                mid = (low + high) // 2
                try:
                    # Test with random input
                    test_input = torch.randn(mid, *sample_input_shape, device=self.device)
                    _ = model(test_input)
                    torch.cuda.synchronize()
                    
                    # If successful, try larger
                    optimal = mid
                    low = mid + 1
                    
                    # Clear for next iteration
                    del test_input
                    self.clear_cache()
                    
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        high = mid - 1
                        self.clear_cache()
                    else:
                        raise e
                        
        print(f"Optimal batch size: {optimal}")
        return optimal
        
    def enable_mixed_precision(self) -> bool:
        """Check if mixed precision is supported and enable it"""
        if not self.is_cuda:
            return False
            
        # Check for Tensor Core support (Volta and newer)
        compute_capability = torch.cuda.get_device_capability()
        supports_mixed_precision = compute_capability[0] >= 7
        
        if supports_mixed_precision:
            print("Mixed precision training enabled (FP16)")
            return True
        else:
            warnings.warn("Mixed precision not supported on this GPU")
            return False
            
    def get_device_summary(self) -> str:
        """Get formatted device summary"""
        if not self.is_cuda:
            return "Device: CPU"
            
        props = self.device_properties
        summary = [
            f"Device: {props['name']}",
            f"Memory: {props['total_memory_gb']:.1f} GB",
            f"SMs: {props['multiprocessor_count']}",
            f"CUDA Cores: ~{props['cuda_cores']}",
            f"Tensor Cores: {props['tensor_cores']}",
            f"Memory Bandwidth: {props['memory_bandwidth_gb']} GB/s",
            f"Mixed Precision: {'Yes' if props['supports_fp16'] else 'No'}",
            f"BF16 Support: {'Yes' if props['supports_bf16'] else 'No'}"
        ]
        return "\n".join(summary)