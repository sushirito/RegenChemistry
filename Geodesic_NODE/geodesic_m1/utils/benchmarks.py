"""Benchmarking utilities for M1 Mac performance testing"""

import torch
import time
from typing import Dict, Any

class M1Benchmarks:
    """Benchmarking tools for M1 Mac geodesic NODE"""
    
    def __init__(self, device: torch.device = torch.device('mps')):
        self.device = device
        
    def benchmark_basic_ops(self) -> Dict[str, float]:
        """Benchmark basic tensor operations"""
        results = {}
        
        # Matrix multiplication
        a = torch.randn(1000, 1000, device=self.device)
        b = torch.randn(1000, 1000, device=self.device)
        
        start = time.perf_counter()
        for _ in range(10):
            c = torch.matmul(a, b)
        end = time.perf_counter()
        
        results['matmul_time'] = (end - start) / 10
        
        return results