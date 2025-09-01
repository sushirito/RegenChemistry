#!/usr/bin/env python3
"""
MPS (Metal Performance Shaders) Utilities
Optimized device management for M1/M3 Mac GPUs
"""

import torch
import platform
import subprocess
from typing import Optional, Dict, Tuple
import warnings


class MPSDeviceManager:
    """Manages MPS device configuration and optimization"""
    
    def __init__(self):
        self.device = self._setup_device()
        self.device_info = self._get_device_info()
        self.optimal_batch_size = self._estimate_optimal_batch_size()
        
    def _setup_device(self) -> torch.device:
        """Setup the optimal device (MPS > CPU)"""
        if torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                warnings.warn("MPS backend is available but not built. Falling back to CPU.")
                return torch.device('cpu')
            return torch.device('mps')
        return torch.device('cpu')
    
    def _get_device_info(self) -> Dict[str, any]:
        """Get device information"""
        info = {
            'device_type': str(self.device),
            'platform': platform.platform(),
            'processor': platform.processor(),
        }
        
        if self.device.type == 'mps':
            # Get Mac system info
            try:
                # Get chip info
                chip_info = subprocess.check_output(
                    ['sysctl', '-n', 'machdep.cpu.brand_string'], 
                    text=True
                ).strip()
                info['chip'] = chip_info
                
                # Estimate GPU cores based on chip
                if 'M1' in chip_info:
                    if 'Max' in chip_info:
                        info['gpu_cores'] = 32
                        info['memory_gb'] = 32  # Minimum for M1 Max
                    elif 'Pro' in chip_info:
                        info['gpu_cores'] = 16
                        info['memory_gb'] = 16  # Minimum for M1 Pro
                    else:
                        info['gpu_cores'] = 8
                        info['memory_gb'] = 8  # M1 base
                elif 'M2' in chip_info:
                    if 'Max' in chip_info:
                        info['gpu_cores'] = 38
                        info['memory_gb'] = 32
                    elif 'Pro' in chip_info:
                        info['gpu_cores'] = 19
                        info['memory_gb'] = 16
                    else:
                        info['gpu_cores'] = 10
                        info['memory_gb'] = 8
                elif 'M3' in chip_info:
                    if 'Max' in chip_info:
                        info['gpu_cores'] = 40
                        info['memory_gb'] = 36
                    elif 'Pro' in chip_info:
                        info['gpu_cores'] = 18
                        info['memory_gb'] = 18
                    else:
                        info['gpu_cores'] = 10
                        info['memory_gb'] = 8
                else:
                    # Default conservative estimates
                    info['gpu_cores'] = 8
                    info['memory_gb'] = 8
                    
            except Exception as e:
                info['error'] = str(e)
                info['gpu_cores'] = 8
                info['memory_gb'] = 8
        
        return info
    
    def _estimate_optimal_batch_size(self) -> int:
        """Estimate optimal batch size based on available memory"""
        if self.device.type == 'mps':
            memory_gb = self.device_info.get('memory_gb', 8)
            gpu_cores = self.device_info.get('gpu_cores', 8)
            
            # Conservative estimates for batch size
            # Reserve 4GB for system and other processes
            available_gb = max(memory_gb - 4, 2)
            
            # Estimate based on model size (~50MB) and data
            # Each sample with 601 wavelengths needs ~10KB
            batch_size = min(
                int(available_gb * 1024 * 100),  # MB to samples
                gpu_cores * 32,  # Scale with GPU cores
                2048  # Maximum reasonable batch
            )
            
            # Round to nearest power of 2 for efficiency
            import math
            batch_size = 2 ** int(math.log2(batch_size))
            return max(batch_size, 32)  # Minimum batch size
        
        return 32  # CPU default
    
    def get_batch_config(self, target_batch: int = 2048) -> Dict[str, int]:
        """Get batch configuration with gradient accumulation"""
        micro_batch = min(self.optimal_batch_size, target_batch)
        accumulation_steps = max(target_batch // micro_batch, 1)
        
        return {
            'micro_batch_size': micro_batch,
            'accumulation_steps': accumulation_steps,
            'effective_batch_size': micro_batch * accumulation_steps,
            'parallel_wavelengths': 601,  # Process all wavelengths
            'concurrent_transitions': 30,  # All concentration pairs
        }
    
    def optimize_model_for_device(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimize model for MPS/CPU execution"""
        model = model.to(self.device)
        
        if self.device.type == 'mps':
            # MPS optimizations
            model.eval()  # Set to eval to enable graph optimization
            
            # Use torch.compile if available (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model, backend='aot_eager')
                except Exception as e:
                    warnings.warn(f"torch.compile failed: {e}")
            
            model.train()  # Set back to train mode
        
        return model
    
    def create_optimal_tensor(self, *args, **kwargs) -> torch.Tensor:
        """Create tensor with optimal settings for device"""
        if 'device' not in kwargs:
            kwargs['device'] = self.device
        
        # Use float32 by default (MPS doesn't fully support float16 yet)
        if 'dtype' not in kwargs:
            kwargs['dtype'] = torch.float32
            
        return torch.tensor(*args, **kwargs)
    
    def synchronize(self):
        """Synchronize device operations"""
        if self.device.type == 'mps':
            torch.mps.synchronize()
        # CPU doesn't need synchronization
    
    def empty_cache(self):
        """Clear device cache"""
        if self.device.type == 'mps':
            torch.mps.empty_cache()
        # CPU doesn't have cache to clear
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics"""
        stats = {}
        
        if self.device.type == 'mps':
            # MPS doesn't expose detailed memory stats yet
            # Use system memory as proxy
            import psutil
            mem = psutil.virtual_memory()
            stats['total_gb'] = mem.total / (1024**3)
            stats['available_gb'] = mem.available / (1024**3)
            stats['used_gb'] = mem.used / (1024**3)
            stats['percent'] = mem.percent
        
        return stats


def parallel_batch_operation(operation, inputs: torch.Tensor, 
                            batch_size: Optional[int] = None) -> torch.Tensor:
    """
    Execute operation in parallel batches optimized for MPS
    
    Args:
        operation: Function to apply
        inputs: Input tensor
        batch_size: Optional batch size override
    
    Returns:
        Output tensor
    """
    if batch_size is None:
        # Process all at once if possible
        return operation(inputs)
    
    # Process in chunks
    outputs = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i+batch_size]
        outputs.append(operation(batch))
    
    return torch.cat(outputs, dim=0)


def enable_mixed_precision_if_available():
    """Enable mixed precision training if supported"""
    # MPS doesn't fully support autocast yet, but we prepare for it
    supports_autocast = False
    
    # Check if MPS supports autocast (future PyTorch versions)
    if torch.backends.mps.is_available():
        try:
            # Test autocast support
            with torch.autocast(device_type='mps', dtype=torch.float16):
                test = torch.randn(2, 2, device='mps')
                _ = test @ test
            supports_autocast = True
        except:
            supports_autocast = False
    
    return supports_autocast


if __name__ == "__main__":
    # Test MPS utilities
    print("Testing MPS Device Manager...")
    
    manager = MPSDeviceManager()
    print(f"\nDevice: {manager.device}")
    print(f"Device Info: {manager.device_info}")
    print(f"Optimal Batch Size: {manager.optimal_batch_size}")
    
    batch_config = manager.get_batch_config(target_batch=2048)
    print(f"\nBatch Configuration:")
    for key, value in batch_config.items():
        print(f"  {key}: {value}")
    
    # Test tensor creation
    test_tensor = manager.create_optimal_tensor([1, 2, 3])
    print(f"\nTest tensor device: {test_tensor.device}")
    print(f"Test tensor dtype: {test_tensor.dtype}")
    
    # Test memory stats
    mem_stats = manager.get_memory_stats()
    if mem_stats:
        print(f"\nMemory Statistics:")
        for key, value in mem_stats.items():
            print(f"  {key}: {value:.2f}")
    
    # Test mixed precision
    supports_amp = enable_mixed_precision_if_available()
    print(f"\nMixed Precision Support: {supports_amp}")