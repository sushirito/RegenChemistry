"""
M1 Mac device and memory management utilities
Optimizes MPS performance and unified memory usage
"""

import torch
import psutil
import platform
from typing import Optional, Dict, Any


class M1DeviceManager:
    """Manages M1 Mac MPS device optimization and memory"""
    
    def __init__(self):
        """Initialize M1 device manager"""
        self.device = self._get_optimal_device()
        self.system_info = self._get_system_info()
        self.memory_stats = {}
        
    def _get_optimal_device(self) -> torch.device:
        """Get the best available device for M1 Mac"""
        if torch.backends.mps.is_available():
            print("✅ MPS acceleration available and enabled")
            return torch.device('mps')
        elif torch.cuda.is_available():
            print("⚠️  CUDA available but preferring CPU for M1 Mac")
            return torch.device('cpu')
        else:
            print("💻 Using CPU (MPS not available)")
            return torch.device('cpu')
            
    def _get_system_info(self) -> Dict[str, Any]:
        """Get M1 Mac system information"""
        return {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'machine': platform.machine(),
            'system': platform.system(),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3),
            'cpu_count': psutil.cpu_count(),
            'mps_available': torch.backends.mps.is_available()
        }
        
    def optimize_for_training(self, batch_size: int = 2048) -> Dict[str, Any]:
        """
        Optimize M1 Mac for training
        
        Args:
            batch_size: Target batch size for optimization
            
        Returns:
            Optimization recommendations
        """
        memory_gb = self.system_info['total_memory_gb']
        
        # Estimate optimal batch size based on memory
        if memory_gb >= 32:
            recommended_batch = min(4096, batch_size)
        elif memory_gb >= 16:
            recommended_batch = min(2048, batch_size)
        else:
            recommended_batch = min(1024, batch_size)
            
        # MPS-specific optimizations
        optimizations = {
            'device': self.device,
            'recommended_batch_size': recommended_batch,
            'use_mixed_precision': self.device.type == 'mps',
            'gradient_accumulation_steps': max(1, batch_size // recommended_batch),
            'memory_efficient_attention': True,
            'pin_memory': False,  # Not needed for unified memory
            'num_workers': min(4, self.system_info['cpu_count'])
        }
        
        print(f"🚀 M1 Mac optimizations:")
        print(f"   Device: {optimizations['device']}")
        print(f"   Recommended batch size: {optimizations['recommended_batch_size']}")
        print(f"   Gradient accumulation: {optimizations['gradient_accumulation_steps']}x")
        print(f"   Mixed precision: {optimizations['use_mixed_precision']}")
        
        return optimizations
        
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics"""
        vm = psutil.virtual_memory()
        
        stats = {
            'total_gb': vm.total / (1024**3),
            'available_gb': vm.available / (1024**3),
            'used_gb': vm.used / (1024**3),
            'usage_percent': vm.percent
        }
        
        # Try to get MPS memory info (if available)
        if self.device.type == 'mps':
            try:
                # MPS memory tracking (approximate)
                stats['mps_allocated_mb'] = torch.mps.current_allocated_memory() / (1024**2)
            except:
                stats['mps_allocated_mb'] = 0.0
                
        self.memory_stats = stats
        return stats
        
    def clear_cache(self):
        """Clear memory caches"""
        if self.device.type == 'mps':
            try:
                torch.mps.empty_cache()
                print("🧹 MPS cache cleared")
            except:
                pass
        elif self.device.type == 'cuda':
            torch.cuda.empty_cache()
            print("🧹 CUDA cache cleared")
            
    def monitor_memory_usage(self, operation_name: str = ""):
        """Monitor memory usage for an operation"""
        stats = self.get_memory_stats()
        
        if operation_name:
            print(f"📊 Memory usage ({operation_name}):")
        else:
            print("📊 Current memory usage:")
            
        print(f"   Total: {stats['total_gb']:.1f} GB")
        print(f"   Used: {stats['used_gb']:.1f} GB ({stats['usage_percent']:.1f}%)")
        print(f"   Available: {stats['available_gb']:.1f} GB")
        
        if 'mps_allocated_mb' in stats:
            print(f"   MPS allocated: {stats['mps_allocated_mb']:.1f} MB")
            
    def find_optimal_batch_size(self, model: torch.nn.Module, 
                               input_shape: tuple, 
                               max_batch: int = 8192) -> int:
        """
        Find optimal batch size for given model and input shape
        
        Args:
            model: PyTorch model to test
            input_shape: Input tensor shape (without batch dimension)
            max_batch: Maximum batch size to test
            
        Returns:
            Optimal batch size
        """
        print(f"🔍 Finding optimal batch size for M1 Mac...")
        model = model.to(self.device)
        model.eval()
        
        optimal_batch = 32
        
        for batch_size in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
            if batch_size > max_batch:
                break
                
            try:
                # Create test input
                test_input = torch.randn(batch_size, *input_shape, device=self.device)
                
                # Test forward pass
                with torch.no_grad():
                    _ = model(test_input)
                    
                # If successful, update optimal batch size
                optimal_batch = batch_size
                print(f"   ✅ Batch size {batch_size}: OK")
                
                # Clear cache for next test
                self.clear_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"   ❌ Batch size {batch_size}: OOM")
                    break
                else:
                    print(f"   ❌ Batch size {batch_size}: Error - {e}")
                    break
                    
        print(f"🎯 Optimal batch size: {optimal_batch}")
        return optimal_batch
        
    def print_system_info(self):
        """Print comprehensive system information"""
        print("🖥️  M1 Mac System Information:")
        print(f"   Platform: {self.system_info['platform']}")
        print(f"   Processor: {self.system_info['processor']}")
        print(f"   Architecture: {self.system_info['machine']}")
        print(f"   Total Memory: {self.system_info['total_memory_gb']:.1f} GB")
        print(f"   CPU Cores: {self.system_info['cpu_count']}")
        print(f"   MPS Available: {self.system_info['mps_available']}")
        print(f"   PyTorch Version: {torch.__version__}")
        
        if self.device.type == 'mps':
            print(f"   🚀 MPS acceleration: ENABLED")
        else:
            print(f"   ⚠️  MPS acceleration: DISABLED")


# Global device manager instance
m1_device_manager = M1DeviceManager()