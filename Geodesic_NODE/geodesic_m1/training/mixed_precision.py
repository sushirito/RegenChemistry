"""
Mixed precision training utilities for M1 Mac
Handles MPS-specific mixed precision with fallback strategies
"""

import torch
import warnings
from typing import Optional, Any, Dict
from contextlib import contextmanager


class M1MixedPrecision:
    """Manages mixed precision training on M1 Mac with MPS fallbacks"""
    
    def __init__(self, 
                 enabled: bool = True,
                 device: torch.device = None,
                 init_scale: float = 2.**16,
                 growth_factor: float = 2.0,
                 backoff_factor: float = 0.5,
                 growth_interval: int = 2000):
        """
        Initialize mixed precision manager for M1 Mac
        
        Args:
            enabled: Whether to use mixed precision
            device: Target device (MPS for M1)
            init_scale: Initial loss scaling factor
            growth_factor: Factor to increase scale when stable
            backoff_factor: Factor to decrease scale on overflow
            growth_interval: Steps between scale increases
        """
        if device is None:
            device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
            
        self.device = device
        self.enabled = enabled and device.type == 'mps'
        
        if self.enabled:
            # MPS doesn't support GradScaler yet, disable scaling
            self.scaler = None
            self.autocast_available = True
            print("✅ MPS acceleration available and enabled")
        else:
            self.scaler = None
            self.autocast_available = False
            if device.type == 'mps':
                print("⚠️  Mixed precision disabled for MPS")
            else:
                print("💻 Using CPU - mixed precision disabled")
                
        self.step_count = 0
        self.overflow_count = 0
        
    @contextmanager
    def autocast(self):
        """Context manager for automatic mixed precision"""
        # MPS doesn't support autocast yet, just pass through
        yield
            
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision training"""
        if self.enabled and self.scaler is not None:
            return self.scaler.scale(loss)
        return loss
        
    def step_optimizer(self, optimizer: torch.optim.Optimizer) -> bool:
        """
        Step optimizer with gradient scaling
        
        Args:
            optimizer: PyTorch optimizer
            
        Returns:
            bool: True if step was taken, False if skipped due to overflow
        """
        self.step_count += 1
        
        if self.enabled and self.scaler is not None:
            # Unscale gradients first
            self.scaler.unscale_(optimizer)
            
            # Check for gradient overflow
            if self._has_gradient_overflow():
                self.overflow_count += 1
                self.scaler.update()
                return False
                
            # Take optimizer step
            optimizer.step()
            self.scaler.update()
            return True
        else:
            # Standard optimizer step
            optimizer.step()
            return True
            
    def _has_gradient_overflow(self) -> bool:
        """Check if gradients have overflowed"""
        if not self.enabled or self.scaler is None:
            return False
            
        # This is a simplified overflow check
        # In practice, GradScaler handles this internally
        return False
        
    def get_scale(self) -> float:
        """Get current loss scale"""
        if self.enabled and self.scaler is not None:
            return self.scaler.get_scale()
        return 1.0
        
    def get_stats(self) -> Dict[str, Any]:
        """Get mixed precision statistics"""
        return {
            'enabled': self.enabled,
            'device': str(self.device),
            'autocast_available': self.autocast_available,
            'current_scale': self.get_scale(),
            'step_count': self.step_count,
            'overflow_count': self.overflow_count,
            'overflow_rate': self.overflow_count / max(1, self.step_count)
        }
        
    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for checkpointing"""
        state = {
            'enabled': self.enabled,
            'step_count': self.step_count,
            'overflow_count': self.overflow_count
        }
        
        if self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()
            
        return state
        
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict from checkpoint"""
        self.step_count = state_dict.get('step_count', 0)
        self.overflow_count = state_dict.get('overflow_count', 0)
        
        if self.scaler is not None and 'scaler' in state_dict:
            self.scaler.load_state_dict(state_dict['scaler'])
            
    def print_status(self):
        """Print current mixed precision status"""
        stats = self.get_stats()
        print("🎯 Mixed Precision Status:")
        print(f"   Enabled: {stats['enabled']}")
        print(f"   Device: {stats['device']}")
        print(f"   Current Scale: {stats['current_scale']:.1f}")
        print(f"   Steps: {stats['step_count']}")
        print(f"   Overflows: {stats['overflow_count']} ({stats['overflow_rate']:.1%})")


def create_m1_mixed_precision(device: Optional[torch.device] = None) -> M1MixedPrecision:
    """
    Factory function to create M1-optimized mixed precision manager
    
    Args:
        device: Target device (auto-detected if None)
        
    Returns:
        Configured M1MixedPrecision instance
    """
    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
    return M1MixedPrecision(
        enabled=device.type == 'mps',
        device=device,
        init_scale=2.**14,  # Conservative initial scale for M1
        growth_factor=1.5,  # Slower growth for stability
        backoff_factor=0.8, # Gentler backoff
        growth_interval=1000  # More frequent scale adjustments
    )