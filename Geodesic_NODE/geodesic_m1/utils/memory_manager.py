"""Memory management utilities for M1 Mac unified memory"""

import torch
from typing import Dict, Any

class M1MemoryManager:
    """Simple memory management for M1 Mac"""
    
    def __init__(self, device: torch.device = torch.device('mps')):
        self.device = device
        
    def clear_cache(self):
        """Clear GPU memory cache"""
        if self.device.type == 'mps':
            try:
                torch.mps.empty_cache()
            except:
                pass