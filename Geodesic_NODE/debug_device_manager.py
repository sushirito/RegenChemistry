"""
M3-Enhanced Device Manager for Local Debugging
Handles CUDA, MPS (Apple Silicon), and CPU with proper fallbacks
"""

import torch
import warnings
from typing import Optional, Dict, Any
import platform


class M3DeviceManager:
    """Enhanced device manager with M3/MPS support for debugging"""
    
    def __init__(self, force_cpu: bool = False, prefer_mps: bool = True):
        """
        Initialize device manager with M3 optimizations
        
        Args:
            force_cpu: Force CPU usage even if GPU/MPS available
            prefer_mps: Prefer MPS over CPU on Apple Silicon
        """
        self.device = self._setup_device(force_cpu, prefer_mps)
        self.device_type = self.device.type
        self.is_cuda = self.device_type == 'cuda'
        self.is_mps = self.device_type == 'mps'
        self.is_apple_silicon = self._is_apple_silicon()
        
        self.device_properties = self._get_device_properties()
        
        # Configure device-specific optimizations
        self._configure_optimizations()
        
    def _is_apple_silicon(self) -> bool:
        """Check if running on Apple Silicon"""
        try:
            # Check if we're on macOS with Apple Silicon
            if platform.system() == 'Darwin':
                import subprocess
                result = subprocess.run(['uname', '-m'], capture_output=True, text=True)
                return 'arm64' in result.stdout
        except:
            pass
        return False
        
    def _setup_device(self, force_cpu: bool, prefer_mps: bool) -> torch.device:
        """Setup device with proper fallback hierarchy"""
        if force_cpu:
            print("🖥️ Forcing CPU usage")
            return torch.device('cpu')
            
        # Priority: CUDA > MPS > CPU
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"🚀 Using CUDA GPU: {device_name}")
            
            if 'A100' in device_name:
                print("✅ A100 detected - full A100 optimizations enabled")
            else:
                print(f"⚠️ Non-A100 GPU detected ({device_name})")
                
            return torch.device('cuda:0')
            
        elif torch.backends.mps.is_available() and prefer_mps:
            print("🍎 Using Apple Silicon MPS")
            if self._is_apple_silicon():
                print("✅ Apple Silicon (M-series) detected")
            return torch.device('mps')
            
        else:
            reasons = []
            if not torch.cuda.is_available():
                reasons.append("CUDA not available")
            if not torch.backends.mps.is_available():
                reasons.append("MPS not available")
                
            print(f"🔄 Falling back to CPU: {', '.join(reasons)}")
            return torch.device('cpu')
    
    def _get_device_properties(self) -> Dict[str, Any]:
        """Get device properties for optimization"""
        base_props = {
            'device_type': self.device_type,
            'is_apple_silicon': self.is_apple_silicon
        }
        
        if self.is_cuda:
            props = torch.cuda.get_device_properties(0)
            return {
                **base_props,
                'name': props.name,
                'total_memory_gb': props.total_memory / (1024**3),
                'multiprocessor_count': props.multi_processor_count,
                'cuda_cores': props.multi_processor_count * 64,
                'memory_bandwidth_gb': 1555 if 'A100' in props.name else 900,
                'supports_fp16': True,
                'supports_bf16': torch.cuda.is_bf16_supported(),
                'tensor_cores': 432 if 'A100' in props.name else 0,
                'optimal_batch_size': 2048 if 'A100' in props.name else 512
            }
            
        elif self.is_mps:
            # MPS properties (estimates for M3)
            return {
                **base_props,
                'name': 'Apple M3 GPU' if self.is_apple_silicon else 'Apple MPS',
                'total_memory_gb': 16.0,  # Unified memory, conservative estimate
                'gpu_cores': 40,  # M3 Pro estimate
                'supports_fp16': True,
                'supports_bf16': False,  # MPS typically doesn't support BF16
                'optimal_batch_size': 256,  # Conservative for unified memory
                'unified_memory': True
            }
            
        else:
            # CPU properties
            return {
                **base_props,
                'name': 'CPU',
                'optimal_batch_size': 64,  # Very conservative for CPU
                'supports_fp16': False,
                'supports_bf16': False
            }
    
    def _configure_optimizations(self):
        """Configure device-specific optimizations"""
        if self.is_cuda:
            # CUDA optimizations
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            if 'A100' in self.device_properties['name']:
                torch.cuda.set_per_process_memory_fraction(0.95)
            else:
                torch.cuda.set_per_process_memory_fraction(0.85)
                
            print("🔧 CUDA optimizations enabled")
            
        elif self.is_mps:
            # MPS optimizations
            print("🔧 MPS optimizations enabled")
            print("   - Using unified memory architecture")
            print("   - FP16 support available")
            
        else:
            # CPU optimizations
            torch.set_num_threads(4)  # Conservative for debugging
            print("🔧 CPU optimizations enabled")
            print(f"   - Using {torch.get_num_threads()} threads")
    
    def get_debug_config(self) -> Dict[str, Any]:
        """Get debug-optimized configuration"""
        base_config = {
            'device': self.device,
            'mixed_precision': self.supports_mixed_precision(),
            'gradient_checkpointing': False,  # Disable for debugging
        }
        
        if self.is_cuda:
            if 'A100' in self.device_properties['name']:
                # A100 debug config
                return {
                    **base_config,
                    'batch_size': 1024,  # Reduced for debugging
                    'christoffel_grid_size': (1000, 301),  # Reduced resolution
                    'n_trajectory_points': 25,
                    'use_tensor_cores': True,
                }
            else:
                # Other CUDA debug config
                return {
                    **base_config,
                    'batch_size': 512,
                    'christoffel_grid_size': (500, 151),
                    'n_trajectory_points': 15,
                    'use_tensor_cores': False,
                }
                
        elif self.is_mps:
            # M3/MPS debug config
            return {
                **base_config,
                'batch_size': 256,  # Conservative for unified memory
                'christoffel_grid_size': (500, 151),  # Reduced for MPS
                'n_trajectory_points': 15,
                'use_tensor_cores': False,
                'mixed_precision': False,  # Often unstable on MPS
            }
            
        else:
            # CPU debug config
            return {
                **base_config,
                'batch_size': 64,
                'christoffel_grid_size': (200, 61),  # Very small for CPU
                'n_trajectory_points': 10,
                'use_tensor_cores': False,
                'mixed_precision': False,
            }
    
    def supports_mixed_precision(self) -> bool:
        """Check if mixed precision is supported"""
        if self.is_cuda:
            compute_capability = torch.cuda.get_device_capability()
            return compute_capability[0] >= 7
        elif self.is_mps:
            return True  # MPS supports FP16
        else:
            return False
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get memory usage information"""
        if self.is_cuda:
            return {
                'allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                'total_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3)
            }
        elif self.is_mps:
            return {
                'allocated_gb': torch.mps.current_allocated_memory() / (1024**3) if hasattr(torch.mps, 'current_allocated_memory') else 0.0,
                'total_gb': 16.0,  # Unified memory estimate
            }
        else:
            return {'mode': 'CPU - memory managed by OS'}
    
    def clear_cache(self):
        """Clear device cache"""
        if self.is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif self.is_mps:
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
    
    def get_device_summary(self) -> str:
        """Get formatted device summary for debugging"""
        props = self.device_properties
        
        summary_lines = [
            f"🖥️ Device: {props['name']}",
            f"🔧 Type: {self.device_type.upper()}",
        ]
        
        if 'total_memory_gb' in props:
            summary_lines.append(f"💾 Memory: {props['total_memory_gb']:.1f} GB")
            
        if 'optimal_batch_size' in props:
            summary_lines.append(f"📦 Optimal batch size: {props['optimal_batch_size']}")
            
        summary_lines.extend([
            f"⚡ Mixed Precision: {'Yes' if self.supports_mixed_precision() else 'No'}",
            f"🍎 Apple Silicon: {'Yes' if self.is_apple_silicon else 'No'}"
        ])
        
        return "\n".join(summary_lines)