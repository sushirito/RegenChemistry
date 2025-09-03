"""
Memory-efficient caching for M1 Mac unified memory
Manages Christoffel grids, trajectories, and intermediate results
"""

import torch
from typing import Dict, Any, Optional, Tuple, Union
import weakref
import gc
from pathlib import Path
import pickle
import hashlib


class M1CacheManager:
    """Manages caching for M1 Mac with unified memory architecture"""
    
    def __init__(self,
                 device: torch.device = torch.device('mps'),
                 max_memory_mb: float = 8192,  # 8GB cache limit
                 disk_cache_dir: Optional[str] = None):
        """
        Initialize M1 cache manager
        
        Args:
            device: Target device (MPS for M1)
            max_memory_mb: Maximum memory to use for caching (MB)
            disk_cache_dir: Directory for disk-based caching
        """
        self.device = device
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.current_memory_usage = 0
        
        # In-memory cache
        self.cache = {}
        self.cache_metadata = {}
        self.access_order = []  # For LRU eviction
        
        # Disk cache setup
        if disk_cache_dir:
            self.disk_cache_dir = Path(disk_cache_dir)
            self.disk_cache_dir.mkdir(exist_ok=True, parents=True)
        else:
            self.disk_cache_dir = None
            
        # Weak references for automatic cleanup
        self.weak_refs = {}
        
    def _get_cache_key(self, 
                      identifier: str, 
                      params: Optional[Dict[str, Any]] = None) -> str:
        """Generate unique cache key from identifier and parameters"""
        if params:
            # Create hash from parameters for uniqueness
            param_str = str(sorted(params.items()))
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
            return f"{identifier}_{param_hash}"
        return identifier
        
    def _get_tensor_memory_usage(self, tensor: torch.Tensor) -> int:
        """Calculate memory usage of tensor in bytes"""
        return tensor.element_size() * tensor.nelement()
        
    def _update_access_order(self, key: str):
        """Update LRU access order"""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        
    def _evict_lru(self, bytes_needed: int):
        """Evict least recently used items to free memory"""
        while (self.current_memory_usage + bytes_needed > self.max_memory_bytes 
               and self.access_order):
            
            lru_key = self.access_order.pop(0)
            
            if lru_key in self.cache:
                # Get memory usage before deletion
                cached_item = self.cache[lru_key]
                if isinstance(cached_item, torch.Tensor):
                    freed_bytes = self._get_tensor_memory_usage(cached_item)
                elif isinstance(cached_item, dict):
                    freed_bytes = sum(
                        self._get_tensor_memory_usage(v) for v in cached_item.values()
                        if isinstance(v, torch.Tensor)
                    )
                else:
                    freed_bytes = 1024  # Estimate for non-tensor objects
                
                # Remove from cache
                del self.cache[lru_key]
                del self.cache_metadata[lru_key]
                self.current_memory_usage -= freed_bytes
                
                print(f"🗑️  Evicted {lru_key} (freed {freed_bytes/1024/1024:.1f} MB)")
                
    def store_christoffel_grid(self,
                             grid: torch.Tensor,
                             grid_params: Dict[str, Any]) -> str:
        """
        Store Christoffel symbol grid in cache
        
        Args:
            grid: Christoffel grid tensor
            grid_params: Grid parameters for key generation
            
        Returns:
            Cache key for retrieval
        """
        cache_key = self._get_cache_key("christoffel_grid", grid_params)
        grid_memory = self._get_tensor_memory_usage(grid)
        
        # Evict if necessary
        self._evict_lru(grid_memory)
        
        # Store in cache
        self.cache[cache_key] = grid.detach().clone()
        self.cache_metadata[cache_key] = {
            'type': 'christoffel_grid',
            'memory_bytes': grid_memory,
            'grid_params': grid_params,
            'shape': tuple(grid.shape),
            'dtype': str(grid.dtype)
        }
        self.current_memory_usage += grid_memory
        self._update_access_order(cache_key)
        
        print(f"💾 Cached Christoffel grid: {cache_key} "
              f"({grid_memory/1024/1024:.1f} MB)")
              
        return cache_key
        
    def get_christoffel_grid(self, 
                           grid_params: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Retrieve Christoffel grid from cache"""
        cache_key = self._get_cache_key("christoffel_grid", grid_params)
        
        if cache_key in self.cache:
            self._update_access_order(cache_key)
            return self.cache[cache_key]
            
        return None
        
    def store_trajectories(self,
                         trajectories: torch.Tensor,
                         trajectory_id: str,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store geodesic trajectories in cache
        
        Args:
            trajectories: Trajectory tensor [n_time, batch_size, state_dim]
            trajectory_id: Unique identifier for trajectories
            metadata: Optional metadata
            
        Returns:
            Cache key
        """
        cache_key = self._get_cache_key(f"trajectories_{trajectory_id}")
        traj_memory = self._get_tensor_memory_usage(trajectories)
        
        # Check if we should cache (don't cache very large trajectory sets)
        if traj_memory > self.max_memory_bytes * 0.5:
            print(f"⚠️  Trajectory too large for cache: {traj_memory/1024/1024:.1f} MB")
            return ""
            
        # Evict if necessary
        self._evict_lru(traj_memory)
        
        # Store in cache
        self.cache[cache_key] = trajectories.detach().clone()
        self.cache_metadata[cache_key] = {
            'type': 'trajectories',
            'memory_bytes': traj_memory,
            'shape': tuple(trajectories.shape),
            'metadata': metadata or {}
        }
        self.current_memory_usage += traj_memory
        self._update_access_order(cache_key)
        
        return cache_key
        
    def get_trajectories(self, trajectory_id: str) -> Optional[torch.Tensor]:
        """Retrieve trajectories from cache"""
        cache_key = self._get_cache_key(f"trajectories_{trajectory_id}")
        
        if cache_key in self.cache:
            self._update_access_order(cache_key)
            return self.cache[cache_key]
            
        return None
        
    def store_batch_results(self,
                          results: Dict[str, torch.Tensor],
                          batch_id: str) -> str:
        """
        Store batch processing results
        
        Args:
            results: Dictionary of result tensors
            batch_id: Unique batch identifier
            
        Returns:
            Cache key
        """
        cache_key = self._get_cache_key(f"batch_results_{batch_id}")
        
        # Calculate total memory usage
        total_memory = sum(
            self._get_tensor_memory_usage(tensor)
            for tensor in results.values()
            if isinstance(tensor, torch.Tensor)
        )
        
        # Don't cache very large batch results
        if total_memory > self.max_memory_bytes * 0.3:
            return ""
            
        # Evict if necessary
        self._evict_lru(total_memory)
        
        # Store results
        cached_results = {
            k: v.detach().clone() if isinstance(v, torch.Tensor) else v
            for k, v in results.items()
        }
        
        self.cache[cache_key] = cached_results
        self.cache_metadata[cache_key] = {
            'type': 'batch_results',
            'memory_bytes': total_memory,
            'batch_id': batch_id
        }
        self.current_memory_usage += total_memory
        self._update_access_order(cache_key)
        
        return cache_key
        
    def get_batch_results(self, batch_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """Retrieve batch results from cache"""
        cache_key = self._get_cache_key(f"batch_results_{batch_id}")
        
        if cache_key in self.cache:
            self._update_access_order(cache_key)
            return self.cache[cache_key]
            
        return None
        
    def store_model_state(self,
                         model_state: Dict[str, Any],
                         model_id: str) -> str:
        """Store intermediate model states"""
        cache_key = self._get_cache_key(f"model_state_{model_id}")
        
        # Estimate memory usage (rough)
        estimated_memory = sum(
            tensor.element_size() * tensor.nelement()
            for tensor in model_state.values()
            if isinstance(tensor, torch.Tensor)
        )
        
        # Don't cache model states if too large
        if estimated_memory > self.max_memory_bytes * 0.4:
            return ""
            
        self._evict_lru(estimated_memory)
        
        self.cache[cache_key] = model_state
        self.cache_metadata[cache_key] = {
            'type': 'model_state',
            'memory_bytes': estimated_memory,
            'model_id': model_id
        }
        self.current_memory_usage += estimated_memory
        self._update_access_order(cache_key)
        
        return cache_key
        
    def clear_cache(self, cache_type: Optional[str] = None):
        """
        Clear cache entries
        
        Args:
            cache_type: Specific type to clear (None for all)
        """
        keys_to_remove = []
        
        for key, metadata in self.cache_metadata.items():
            if cache_type is None or metadata['type'] == cache_type:
                keys_to_remove.append(key)
                
        freed_memory = 0
        for key in keys_to_remove:
            if key in self.cache_metadata:
                freed_memory += self.cache_metadata[key]['memory_bytes']
                del self.cache[key]
                del self.cache_metadata[key]
                if key in self.access_order:
                    self.access_order.remove(key)
                    
        self.current_memory_usage -= freed_memory
        
        if keys_to_remove:
            print(f"🧹 Cleared {len(keys_to_remove)} cache entries "
                  f"(freed {freed_memory/1024/1024:.1f} MB)")
                  
        # Force garbage collection
        gc.collect()
        
    def save_to_disk(self, cache_key: str, force: bool = False) -> bool:
        """
        Save cache entry to disk
        
        Args:
            cache_key: Key of item to save
            force: Save even if already on disk
            
        Returns:
            True if saved successfully
        """
        if not self.disk_cache_dir:
            return False
            
        if cache_key not in self.cache:
            return False
            
        disk_path = self.disk_cache_dir / f"{cache_key}.pkl"
        
        if disk_path.exists() and not force:
            return True
            
        try:
            # Save to disk
            with open(disk_path, 'wb') as f:
                cache_data = {
                    'data': self.cache[cache_key],
                    'metadata': self.cache_metadata[cache_key]
                }
                pickle.dump(cache_data, f)
                
            print(f"💽 Saved {cache_key} to disk")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save {cache_key} to disk: {e}")
            return False
            
    def load_from_disk(self, cache_key: str) -> bool:
        """
        Load cache entry from disk
        
        Args:
            cache_key: Key of item to load
            
        Returns:
            True if loaded successfully
        """
        if not self.disk_cache_dir:
            return False
            
        disk_path = self.disk_cache_dir / f"{cache_key}.pkl"
        
        if not disk_path.exists():
            return False
            
        try:
            with open(disk_path, 'rb') as f:
                cache_data = pickle.load(f)
                
            # Check memory before loading
            memory_needed = cache_data['metadata']['memory_bytes']
            self._evict_lru(memory_needed)
            
            # Load into cache
            self.cache[cache_key] = cache_data['data']
            self.cache_metadata[cache_key] = cache_data['metadata']
            self.current_memory_usage += memory_needed
            self._update_access_order(cache_key)
            
            print(f"📂 Loaded {cache_key} from disk")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load {cache_key} from disk: {e}")
            return False
            
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        stats = {
            'total_entries': len(self.cache),
            'memory_usage_mb': self.current_memory_usage / (1024 * 1024),
            'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
            'memory_utilization': self.current_memory_usage / self.max_memory_bytes,
            'device': str(self.device),
            'entries_by_type': {}
        }
        
        # Count entries by type
        for metadata in self.cache_metadata.values():
            cache_type = metadata['type']
            if cache_type not in stats['entries_by_type']:
                stats['entries_by_type'][cache_type] = {'count': 0, 'memory_mb': 0}
            
            stats['entries_by_type'][cache_type]['count'] += 1
            stats['entries_by_type'][cache_type]['memory_mb'] += metadata['memory_bytes'] / (1024 * 1024)
            
        return stats
        
    def optimize_cache(self):
        """Optimize cache by removing duplicates and cleaning up"""
        print("🔧 Optimizing cache...")
        
        # Force garbage collection first
        gc.collect()
        
        # Remove any None entries
        none_keys = [k for k, v in self.cache.items() if v is None]
        for key in none_keys:
            self.remove_from_cache(key)
            
        # Save frequently used items to disk if disk cache available
        if self.disk_cache_dir:
            # Save LRU items to disk to free memory
            while (self.current_memory_usage > self.max_memory_bytes * 0.8 
                   and len(self.access_order) > 5):
                lru_key = self.access_order[0]
                if self.save_to_disk(lru_key):
                    self.remove_from_cache(lru_key)
                else:
                    break
                    
        print(f"✅ Cache optimization complete")
        
    def remove_from_cache(self, cache_key: str):
        """Remove specific item from cache"""
        if cache_key in self.cache:
            memory_freed = self.cache_metadata[cache_key]['memory_bytes']
            del self.cache[cache_key]
            del self.cache_metadata[cache_key]
            if cache_key in self.access_order:
                self.access_order.remove(cache_key)
            self.current_memory_usage -= memory_freed
            
    def __del__(self):
        """Cleanup on deletion"""
        self.clear_cache()