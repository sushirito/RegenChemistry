"""Performance profiling utilities for M1 Mac geodesic NODE"""

import torch
import time
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
from collections import defaultdict


class M1Profiler:
    """Performance profiler optimized for M1 Mac MPS"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.timings = defaultdict(list)
        self.current_timings = {}
        self.memory_snapshots = []
        
    @contextmanager
    def profile(self, operation_name: str):
        if not self.enabled:
            yield
            return
            
        start_time = time.perf_counter()
        
        # Memory before
        memory_before = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            # Memory after
            memory_after = self._get_memory_usage()
            
            self.timings[operation_name].append({
                'duration': duration,
                'memory_before': memory_before,
                'memory_after': memory_after,
                'memory_delta': memory_after - memory_before
            })
            
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            return psutil.virtual_memory().used / (1024**2)
        except:
            return 0.0
            
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {}
        
        for operation, timings in self.timings.items():
            durations = [t['duration'] for t in timings]
            memory_deltas = [t['memory_delta'] for t in timings]
            
            summary[operation] = {
                'count': len(durations),
                'total_time': sum(durations),
                'avg_time': sum(durations) / len(durations) if durations else 0,
                'min_time': min(durations) if durations else 0,
                'max_time': max(durations) if durations else 0,
                'avg_memory_delta': sum(memory_deltas) / len(memory_deltas) if memory_deltas else 0
            }
            
        return summary