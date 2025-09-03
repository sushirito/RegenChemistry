"""
Vectorized Christoffel symbol computation with pre-computed grid
Optimized for A100 massive parallelization
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import time


class ChristoffelComputer:
    """Computes and caches Christoffel symbols for geodesic computation"""
    
    def __init__(self, 
                 metric_network: torch.nn.Module,
                 grid_size: Tuple[int, int] = (2000, 601),
                 device: torch.device = torch.device('cuda'),
                 use_half_precision: bool = True):
        """
        Initialize Christoffel computer with pre-computed grid
        
        Args:
            metric_network: Neural network computing g(c,λ)
            grid_size: (concentration_points, wavelength_points)
            device: Computation device
            use_half_precision: Store grid in FP16 for memory efficiency
        """
        self.metric_network = metric_network
        self.grid_size = grid_size
        self.device = device
        self.use_half_precision = use_half_precision
        
        # Grid will be computed on first use or explicitly
        self.christoffel_grid: Optional[torch.Tensor] = None
        self.c_grid: Optional[torch.Tensor] = None
        self.lambda_grid: Optional[torch.Tensor] = None
        
    def precompute_grid(self, c_range: Tuple[float, float] = (-1, 1),
                       lambda_range: Tuple[float, float] = (-1, 1)) -> None:
        """
        Pre-compute Christoffel symbols on dense grid
        Single massive forward pass for efficiency
        
        Args:
            c_range: Normalized concentration range
            lambda_range: Normalized wavelength range
        """
        print(f"Pre-computing Christoffel grid {self.grid_size}...")
        start_time = time.time()
        
        # Create grid points
        c_points = torch.linspace(c_range[0], c_range[1], self.grid_size[0], device=self.device)
        lambda_points = torch.linspace(lambda_range[0], lambda_range[1], self.grid_size[1], device=self.device)
        
        # Store grid coordinates for interpolation
        self.c_grid = c_points
        self.lambda_grid = lambda_points
        
        # Create meshgrid for vectorized computation
        c_mesh, lambda_mesh = torch.meshgrid(c_points, lambda_points, indexing='ij')
        
        # Flatten for batch processing
        c_flat = c_mesh.flatten()
        lambda_flat = lambda_mesh.flatten()
        
        # Compute Christoffel symbols using finite differences
        christoffel_flat = self._compute_christoffel_batch(c_flat, lambda_flat)
        
        # Reshape to grid
        self.christoffel_grid = christoffel_flat.view(self.grid_size[0], self.grid_size[1])
        
        # Convert to half precision if requested (saves memory)
        if self.use_half_precision:
            self.christoffel_grid = self.christoffel_grid.half()
            
        compute_time = time.time() - start_time
        memory_mb = self.christoffel_grid.element_size() * self.christoffel_grid.nelement() / (1024**2)
        
        print(f"  Computation time: {compute_time:.2f}s")
        print(f"  Grid memory: {memory_mb:.2f} MB")
        print(f"  Grid range: Γ ∈ [{self.christoffel_grid.min():.4f}, {self.christoffel_grid.max():.4f}]")
        
    def _compute_christoffel_batch(self, c_batch: torch.Tensor, 
                                  lambda_batch: torch.Tensor) -> torch.Tensor:
        """
        Compute Christoffel symbols for batch of points
        Γ = ½ g⁻¹ ∂g/∂c
        
        Args:
            c_batch: Concentration values [batch_size]
            lambda_batch: Wavelength values [batch_size]
            
        Returns:
            Christoffel symbols [batch_size]
        """
        eps = 1e-4  # Finite difference epsilon
        
        # Prepare perturbed inputs for finite differences
        c_plus = c_batch + eps
        c_minus = c_batch - eps
        
        # Stack inputs for single forward pass
        c_stacked = torch.cat([c_minus, c_batch, c_plus])
        lambda_stacked = torch.cat([lambda_batch, lambda_batch, lambda_batch])
        inputs_stacked = torch.stack([c_stacked, lambda_stacked], dim=1)
        
        # Single forward pass through metric network
        with torch.no_grad():
            g_values = self.metric_network(inputs_stacked)
            
        # Split results
        batch_size = c_batch.shape[0]
        g_minus = g_values[:batch_size]
        g_center = g_values[batch_size:2*batch_size]
        g_plus = g_values[2*batch_size:]
        
        # Compute derivative using central differences
        dg_dc = (g_plus - g_minus) / (2 * eps)
        
        # Compute Christoffel symbol
        # Add small epsilon to avoid division by zero
        christoffel = 0.5 * dg_dc / (g_center + 1e-10)
        
        return christoffel.squeeze()
        
    def interpolate(self, c: torch.Tensor, lambda_val: torch.Tensor) -> torch.Tensor:
        """
        Interpolate Christoffel symbols from pre-computed grid
        Uses bilinear interpolation for smooth values
        
        Args:
            c: Concentration values (normalized) [batch_size]
            lambda_val: Wavelength values (normalized) [batch_size]
            
        Returns:
            Interpolated Christoffel symbols [batch_size]
        """
        if self.christoffel_grid is None:
            raise RuntimeError("Christoffel grid not computed. Call precompute_grid() first.")
            
        batch_size = c.shape[0]
        
        # Convert to grid coordinates [-1, 1] -> [0, grid_size-1]
        c_norm = (c - self.c_grid[0]) / (self.c_grid[-1] - self.c_grid[0])
        lambda_norm = (lambda_val - self.lambda_grid[0]) / (self.lambda_grid[-1] - self.lambda_grid[0])
        
        # Convert to grid indices
        c_idx = c_norm * (self.grid_size[0] - 1)
        lambda_idx = lambda_norm * (self.grid_size[1] - 1)
        
        # Stack for grid_sample (expects [N, 2] for 2D grid)
        # Note: grid_sample expects coordinates in (x,y) = (width, height) order
        # So we need (lambda, c) not (c, lambda)
        grid_coords = torch.stack([lambda_idx * 2 - 1, c_idx * 2 - 1], dim=1)
        
        # Reshape grid for grid_sample [N, 1, H, W] - repeat for batch
        grid = self.christoffel_grid.float().unsqueeze(0).unsqueeze(0)
        grid = grid.expand(batch_size, 1, self.grid_size[0], self.grid_size[1])
        
        # Reshape coordinates for grid_sample [N, 1, 1, 2]
        grid_coords = grid_coords.view(batch_size, 1, 1, 2)
        
        # Bilinear interpolation - use 'zeros' for MPS compatibility
        interpolated = F.grid_sample(grid, grid_coords, 
                                    mode='bilinear', 
                                    padding_mode='zeros',
                                    align_corners=True)
        
        # Reshape output [N, 1, 1, 1] -> [N]
        return interpolated.squeeze()
        
    def get_grid_stats(self) -> dict:
        """Get statistics about the pre-computed grid"""
        if self.christoffel_grid is None:
            return {"status": "Grid not computed"}
            
        return {
            "grid_size": self.grid_size,
            "memory_mb": self.christoffel_grid.element_size() * self.christoffel_grid.nelement() / (1024**2),
            "dtype": str(self.christoffel_grid.dtype),
            "min_value": float(self.christoffel_grid.min()),
            "max_value": float(self.christoffel_grid.max()),
            "mean_value": float(self.christoffel_grid.mean()),
            "std_value": float(self.christoffel_grid.std())
        }