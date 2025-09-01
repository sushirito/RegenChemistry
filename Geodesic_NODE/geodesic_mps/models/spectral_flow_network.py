#!/usr/bin/env python3
"""
Spectral Flow Network for Coupled NODE
Maps (c, v, λ) to dA/dt (rate of absorbance change)
"""

import torch
import torch.nn as nn
from typing import Optional


class SpectralFlowNetwork(nn.Module):
    """
    Ultra-small network for learning spectral dynamics
    Maps current state to rate of absorbance change
    """
    
    def __init__(self, hidden_dim: int = 16, n_layers: int = 1):
        """
        Initialize spectral flow network
        
        Args:
            hidden_dim: Hidden layer dimension (small to prevent overfitting)
            n_layers: Number of hidden layers (typically 1)
        """
        super().__init__()
        
        # Ultra-small architecture: 3 → 16 → 1
        layers = []
        
        # Input layer (c, v, λ)
        layers.append(nn.Linear(3, hidden_dim))
        layers.append(nn.Tanh())
        
        # Hidden layers (if any)
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        # Output layer (dA/dt)
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights for stability
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, c: torch.Tensor, v: torch.Tensor, 
                wavelength: torch.Tensor) -> torch.Tensor:
        """
        Compute rate of absorbance change dA/dt
        
        Args:
            c: Concentration values (batch,)
            v: Velocity values dc/dt (batch,)
            wavelength: Wavelength values (batch,)
        
        Returns:
            dA/dt values (batch,)
        """
        # Ensure proper shapes
        if c.dim() == 0:
            c = c.unsqueeze(0)
        if v.dim() == 0:
            v = v.unsqueeze(0)
        if wavelength.dim() == 0:
            wavelength = wavelength.unsqueeze(0)
        
        # Stack inputs
        x = torch.stack([c, v, wavelength], dim=-1)  # (batch, 3)
        
        # Forward pass
        dA_dt = self.network(x).squeeze(-1)  # (batch,)
        
        return dA_dt
    
    def forward_batch(self, state: torch.Tensor, wavelengths: torch.Tensor) -> torch.Tensor:
        """
        Batch forward for ODE integration
        
        Args:
            state: State tensor (batch, 3) containing [c, v, A]
            wavelengths: Wavelength values (batch,)
        
        Returns:
            dA/dt values (batch,)
        """
        c = state[:, 0]
        v = state[:, 1]
        # Note: state[:, 2] is A, but we don't use it for computing dA/dt
        
        return self.forward(c, v, wavelengths)


class OptimizedSpectralFlow(SpectralFlowNetwork):
    """
    Optimized version with wavelength embedding
    """
    
    def __init__(self, hidden_dim: int = 16, n_wavelengths: int = 601,
                 embedding_dim: int = 8):
        """
        Initialize with wavelength embeddings
        
        Args:
            hidden_dim: Hidden layer dimension
            n_wavelengths: Number of discrete wavelengths
            embedding_dim: Dimension of wavelength embedding
        """
        super().__init__(hidden_dim, n_layers=1)
        
        # Wavelength embedding for better representation
        self.wavelength_embedding = nn.Embedding(n_wavelengths, embedding_dim)
        
        # Rebuild network with embedding
        self.state_encoder = nn.Linear(2, hidden_dim // 2)  # For c, v
        self.wavelength_encoder = nn.Linear(embedding_dim, hidden_dim // 2)
        
        self.hidden = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._initialize_weights()
    
    def forward_with_embedding(self, c: torch.Tensor, v: torch.Tensor,
                              wavelength_idx: torch.Tensor) -> torch.Tensor:
        """
        Forward with wavelength indices instead of normalized values
        
        Args:
            c: Concentration values (batch,)
            v: Velocity values (batch,)
            wavelength_idx: Wavelength indices (batch,) in [0, n_wavelengths)
        
        Returns:
            dA/dt values (batch,)
        """
        # Encode state
        state = torch.stack([c, v], dim=-1)
        state_features = self.state_encoder(state)
        
        # Encode wavelength
        wl_embedding = self.wavelength_embedding(wavelength_idx)
        wl_features = self.wavelength_encoder(wl_embedding)
        
        # Combine and process
        combined = torch.cat([state_features, wl_features], dim=-1)
        dA_dt = self.hidden(combined).squeeze(-1)
        
        return dA_dt


class ParallelSpectralFlow(nn.Module):
    """
    Massively parallel spectral flow for batch processing
    """
    
    def __init__(self, hidden_dim: int = 11):
        """
        Ultra-lightweight for maximum speed
        """
        super().__init__()
        
        # Minimal architecture for speed
        self.fc1 = nn.Linear(3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        # Pre-compile if available
        if hasattr(torch, 'compile'):
            try:
                self.forward = torch.compile(self.forward, backend='aot_eager')
            except:
                pass
    
    def forward(self, c: torch.Tensor, v: torch.Tensor,
                wavelength: torch.Tensor) -> torch.Tensor:
        """
        Fast forward pass
        """
        # Stack and process in one go
        x = torch.stack([c, v, wavelength], dim=-1)
        h = torch.tanh(self.fc1(x))
        dA_dt = self.fc2(h).squeeze(-1)
        return dA_dt


def test_spectral_flow():
    """Test spectral flow network"""
    print("Testing Spectral Flow Network...")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create network
    flow = SpectralFlowNetwork().to(device)
    print(f"Network created with {sum(p.numel() for p in flow.parameters())} parameters")
    
    # Test single forward
    c = torch.tensor([0.5], device=device)
    v = torch.tensor([1.0], device=device)
    wl = torch.tensor([-0.2], device=device)
    
    dA_dt = flow(c, v, wl)
    print(f"\nSingle forward: dA/dt = {dA_dt.item():.4f}")
    
    # Test batch forward
    batch_size = 256
    c_batch = torch.randn(batch_size, device=device)
    v_batch = torch.randn(batch_size, device=device)
    wl_batch = torch.randn(batch_size, device=device)
    
    dA_dt_batch = flow(c_batch, v_batch, wl_batch)
    print(f"\nBatch forward: input {batch_size} → output {dA_dt_batch.shape}")
    print(f"Output range: [{dA_dt_batch.min().item():.4f}, {dA_dt_batch.max().item():.4f}]")
    
    # Test gradient flow
    loss = dA_dt_batch.mean()
    loss.backward()
    
    has_grad = all(p.grad is not None for p in flow.parameters() if p.requires_grad)
    print(f"Gradient flow: {has_grad}")
    
    # Test optimized version
    print("\nTesting optimized spectral flow...")
    opt_flow = OptimizedSpectralFlow(n_wavelengths=601).to(device)
    print(f"Optimized network: {sum(p.numel() for p in opt_flow.parameters())} parameters")
    
    # Test with wavelength indices
    wl_idx = torch.randint(0, 601, (batch_size,), device=device)
    dA_dt_opt = opt_flow.forward_with_embedding(c_batch, v_batch, wl_idx)
    print(f"Optimized output shape: {dA_dt_opt.shape}")
    
    # Test parallel version
    print("\nTesting parallel spectral flow...")
    parallel_flow = ParallelSpectralFlow().to(device)
    print(f"Parallel network: {sum(p.numel() for p in parallel_flow.parameters())} parameters")
    
    dA_dt_parallel = parallel_flow(c_batch, v_batch, wl_batch)
    print(f"Parallel output shape: {dA_dt_parallel.shape}")
    
    # Benchmark
    import time
    n_iterations = 1000
    
    networks = [
        ("Standard", flow),
        ("Optimized", lambda c, v, wl: opt_flow(c, v, wl)),
        ("Parallel", parallel_flow)
    ]
    
    print("\nBenchmarking...")
    for name, model in networks:
        if hasattr(model, 'eval'):
            model.eval()
        
        start = time.time()
        with torch.no_grad():
            for _ in range(n_iterations):
                if callable(model):
                    _ = model(c_batch, v_batch, wl_batch)
                else:
                    _ = model(c_batch, v_batch, wl_batch)
        
        if device.type == 'mps':
            torch.mps.synchronize()
        
        elapsed = time.time() - start
        throughput = n_iterations * batch_size / elapsed
        
        print(f"{name:10s}: {elapsed:.3f}s ({throughput:.0f} samples/s)")
    
    print("\nSpectral flow tests passed!")


if __name__ == "__main__":
    test_spectral_flow()