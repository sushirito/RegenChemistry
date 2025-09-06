"""
M3 Debug Training Script for Geodesic NODE Issues
Systematically debug convergence and smoothness loss problems
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse
import time
from pathlib import Path
import sys
from tqdm import tqdm

# Add the geodesic_a100 directory to path
sys.path.append('geodesic_a100')

from debug_device_manager import M3DeviceManager
from geodesic_a100.models.metric_network import MetricNetwork
from geodesic_a100.models.spectral_flow_network import SpectralFlowNetwork
from geodesic_a100.models.geodesic_model import GeodesicNODE
from geodesic_a100.data.data_loader import SpectralDataset


class DebugTrainingSession:
    """Debug training session with enhanced diagnostics"""
    
    def __init__(self, mode: str = "baseline", data_path: str = None, force_cpu: bool = False):
        self.mode = mode
        self.data_path = data_path or "data/test_spectral_data.csv"
        
        # Setup device
        self.device_manager = M3DeviceManager(force_cpu=force_cpu, prefer_mps=True)
        self.device = self.device_manager.device
        self.config = self.device_manager.get_debug_config()
        
        print("="*60)
        print("🔍 GEODESIC NODE DEBUG SESSION")
        print("="*60)
        print(self.device_manager.get_device_summary())
        print("\n📋 Debug Configuration:")
        for key, value in self.config.items():
            print(f"   {key}: {value}")
        print("="*60)
        
        # Initialize components
        self._setup_model()
        self._setup_data()
        self._setup_training()
        
    def _setup_model(self):
        """Initialize model with debug configuration"""
        print(f"\n🏗️ Setting up model for {self.mode} mode...")
        
        # Get device-specific config
        grid_size = self.config['christoffel_grid_size']
        n_points = self.config['n_trajectory_points']
        
        # Initialize model
        self.model = GeodesicNODE(
            metric_hidden_dims=[64, 128],  # Smaller for debugging
            flow_hidden_dims=[32, 64],     # Smaller for debugging
            n_trajectory_points=n_points,
            shooting_max_iter=15,           # Fewer iterations for debugging
            shooting_tolerance=1e-4,        # Standard tolerance
            shooting_learning_rate=0.5,    # Standard learning rate
            christoffel_grid_size=grid_size,
            device=self.device,
            use_adjoint=False  # Disable for debugging clarity
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"   📊 Total parameters: {total_params:,}")
        print(f"   🎯 Grid size: {grid_size}")
        print(f"   📍 Trajectory points: {n_points}")
        
    def _setup_data(self):
        """Setup data with proper fallbacks"""
        print(f"\n📁 Loading data...")
        
        # Try to load actual data, fallback to synthetic
        try:
            if Path(self.data_path).exists():
                print(f"   Loading from: {self.data_path}")
                # Load real data
                df = pd.read_csv(self.data_path)
                wavelengths = df['Wavelength'].values
                concentrations = [float(col) for col in df.columns[1:]]
                absorbance_data = df.iloc[:, 1:].values.T
                
                self.dataset = SpectralDataset(
                    concentration_values=concentrations,
                    wavelengths=wavelengths,
                    absorbance_data=absorbance_data,
                    excluded_concentration_idx=5,  # Exclude 60 ppb (hardest case)
                    normalize=True,
                    device=self.device
                )
            else:
                raise FileNotFoundError("Data file not found, using synthetic")
                
        except Exception as e:
            print(f"   ⚠️ Could not load real data ({e}), generating synthetic...")
            # Generate synthetic data
            concentrations = [0, 10, 20, 30, 40, 60]
            wavelengths = np.linspace(200, 800, 301)  # Reduced for debugging
            
            self.dataset = SpectralDataset(
                concentration_values=concentrations,
                wavelengths=wavelengths,
                absorbance_data=None,  # Will generate synthetic
                excluded_concentration_idx=5,  # Exclude 60 ppb
                normalize=True,
                device=self.device
            )
        
        print(f"   📊 Dataset size: {len(self.dataset):,} samples")
        print(f"   🎯 Concentrations: {self.dataset.concentration_values}")
        print(f"   📏 Wavelengths: {len(self.dataset.wavelengths)} points")
        
    def _setup_training(self):
        """Setup optimizers and other training components"""
        print(f"\n⚙️ Setting up training components...")
        
        # Optimizers with conservative learning rates for debugging
        self.optimizer_metric = optim.Adam(
            self.model.metric_network.parameters(),
            lr=1e-3,  # Higher for faster debugging
            weight_decay=1e-5
        )
        
        self.optimizer_flow = optim.Adam(
            self.model.spectral_flow_network.parameters(),
            lr=2e-3,  # Higher for faster debugging
            weight_decay=1e-5
        )
        
        # Data loader
        from torch.utils.data import DataLoader
        batch_size = self.config['batch_size']
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        print(f"   📦 Batch size: {batch_size}")
        print(f"   🔄 Batches per epoch: {len(self.dataloader)}")
        
    def run_debug_epoch(self) -> dict:
        """Run a single debug epoch with detailed diagnostics"""
        print(f"\n🚀 Running debug epoch in {self.mode} mode...")
        
        self.model.train()
        epoch_metrics = {
            'total_loss': [],
            'reconstruction_loss': [],
            'smoothness_loss': [],
            'bounds_loss': [],
            'path_length_loss': [],
            'convergence_rates': [],
            'shooting_errors': [],
            'memory_usage': [],
            'batch_times': []
        }
        
        # Pre-compute Christoffel grid with timing
        print("   🔧 Pre-computing Christoffel grid...")
        start_time = time.time()
        self.model.precompute_christoffel_grid()
        grid_time = time.time() - start_time
        print(f"   ⏱️ Grid computation: {grid_time:.2f}s")
        
        # Process batches with detailed diagnostics
        for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Debug Batches")):
            batch_start = time.time()
            
            # Unpack batch
            c_sources, c_targets, wavelengths, target_absorbance = batch
            c_sources = c_sources.to(self.device)
            c_targets = c_targets.to(self.device)
            wavelengths = wavelengths.to(self.device)
            target_absorbance = target_absorbance.to(self.device)
            
            # DETAILED DIAGNOSTICS FOR FIRST BATCH
            if batch_idx == 0:
                self._print_batch_diagnostics(c_sources, c_targets, wavelengths, target_absorbance)
            
            # Zero gradients
            self.optimizer_metric.zero_grad()
            self.optimizer_flow.zero_grad()
            
            try:
                # Forward pass with diagnostic wrapper
                output = self._debug_forward_pass(c_sources, c_targets, wavelengths, batch_idx)
                
                # Loss computation with diagnostics
                loss_dict = self._debug_loss_computation(
                    output, target_absorbance, c_sources, wavelengths, batch_idx
                )
                
                # Backward pass
                loss = loss_dict['total']
                loss.backward()
                
                # Gradient diagnostics
                if batch_idx == 0:
                    self._print_gradient_diagnostics()
                
                # Optimizer step
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer_metric.step()
                self.optimizer_flow.step()
                
                # Record metrics
                epoch_metrics['total_loss'].append(loss.item())
                epoch_metrics['reconstruction_loss'].append(loss_dict['reconstruction'].item())
                epoch_metrics['smoothness_loss'].append(loss_dict['smoothness'].item())
                epoch_metrics['bounds_loss'].append(loss_dict['bounds'].item())
                epoch_metrics['path_length_loss'].append(loss_dict['path_length'].item())
                epoch_metrics['convergence_rates'].append(output.get('convergence_rate', 0).item())
                
                # Memory usage
                memory_info = self.device_manager.get_memory_info()
                if 'allocated_gb' in memory_info:
                    epoch_metrics['memory_usage'].append(memory_info['allocated_gb'])
                
                batch_time = time.time() - batch_start
                epoch_metrics['batch_times'].append(batch_time)
                
            except Exception as e:
                print(f"   ❌ Error in batch {batch_idx}: {e}")
                if batch_idx == 0:  # Critical error in first batch
                    raise e
                continue  # Skip this batch
                
        return self._summarize_epoch_metrics(epoch_metrics)
    
    def _print_batch_diagnostics(self, c_sources, c_targets, wavelengths, target_absorbance):
        """Print detailed diagnostics for first batch"""
        print(f"\n📊 BATCH DIAGNOSTICS (First Batch):")
        print(f"   Batch size: {c_sources.shape[0]}")
        print(f"   c_sources range: [{c_sources.min():.3f}, {c_sources.max():.3f}]")
        print(f"   c_targets range: [{c_targets.min():.3f}, {c_targets.max():.3f}]")
        print(f"   |c_diff| range: [{(c_targets - c_sources).abs().min():.3f}, {(c_targets - c_sources).abs().max():.3f}]")
        print(f"   wavelengths range: [{wavelengths.min():.3f}, {wavelengths.max():.3f}]")
        print(f"   target_absorbance range: [{target_absorbance.min():.3f}, {target_absorbance.max():.3f}]")
        
    def _debug_forward_pass(self, c_sources, c_targets, wavelengths, batch_idx):
        """Forward pass with shooting solver diagnostics"""
        if batch_idx == 0:
            print(f"\n🎯 SHOOTING SOLVER DIAGNOSTICS:")
            
        # Custom forward to capture shooting diagnostics
        bvp_results = self.model.shooting_solver.solve_batch(
            c_sources=c_sources,
            c_targets=c_targets,
            wavelengths=wavelengths,
            n_trajectory_points=self.model.n_trajectory_points
        )
        
        if batch_idx == 0:
            print(f"   Convergence rate: {bvp_results['convergence_rate']:.1%}")
            print(f"   Error range: [{bvp_results['final_errors'].min():.6f}, {bvp_results['final_errors'].max():.6f}]")
            print(f"   Shooting tolerance: {self.model.shooting_solver.tolerance}")
            print(f"   Converged geodesics: {bvp_results['convergence_mask'].sum().item()}/{len(c_sources)}")
            
        return {
            'absorbance': bvp_results['final_absorbance'],
            'trajectories': bvp_results['trajectories'],
            'initial_velocities': bvp_results['initial_velocities'],
            'convergence_mask': bvp_results['convergence_mask'],
            'convergence_rate': bvp_results['convergence_rate']
        }
    
    def _debug_loss_computation(self, output, targets, c_batch, wavelength_batch, batch_idx):
        """Loss computation with smoothness diagnostics"""
        # Reconstruction loss
        reconstruction_loss = nn.functional.mse_loss(output['absorbance'], targets)
        
        if batch_idx == 0:
            print(f"\n📉 LOSS DIAGNOSTICS:")
            print(f"   Reconstruction loss: {reconstruction_loss.item():.6f}")
            
        # DEBUG: Smoothness loss with step-by-step tracking
        try:
            if batch_idx == 0:
                print(f"   Computing smoothness loss for {len(c_batch)} samples...")
            smoothness_loss = self.model.metric_network.get_smoothness_loss(c_batch, wavelength_batch)
            if batch_idx == 0:
                print(f"   Smoothness loss: {smoothness_loss.item():.6f}")
        except Exception as e:
            if batch_idx == 0:
                print(f"   ❌ Smoothness loss failed: {e}")
            smoothness_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
        # Bounds loss
        bounds_loss = self.model.metric_network.get_bounds_loss(c_batch, wavelength_batch)
        
        # Path length loss
        if 'trajectories' in output:
            velocities = output['trajectories'][:, :, 1]
            path_lengths = velocities.abs().mean(dim=0).mean()
        else:
            path_lengths = torch.tensor(0.0, device=self.device)
            
        if batch_idx == 0:
            print(f"   Bounds loss: {bounds_loss.item():.6f}")
            print(f"   Path length loss: {path_lengths.item():.6f}")
            
        # Total loss
        total_loss = (
            reconstruction_loss +
            0.01 * smoothness_loss +
            0.001 * bounds_loss +
            0.001 * path_lengths
        )
        
        return {
            'total': total_loss,
            'reconstruction': reconstruction_loss,
            'smoothness': smoothness_loss,
            'bounds': bounds_loss,
            'path_length': path_lengths
        }
    
    def _print_gradient_diagnostics(self):
        """Print gradient flow diagnostics"""
        print(f"\n🔄 GRADIENT DIAGNOSTICS:")
        
        total_norm = 0
        param_count = 0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                if 'metric' in name:
                    print(f"   {name}: {param_norm:.6f}")
                    
        total_norm = total_norm ** (1. / 2)
        print(f"   Total gradient norm: {total_norm:.6f}")
        print(f"   Parameters with gradients: {param_count}")
    
    def _summarize_epoch_metrics(self, metrics) -> dict:
        """Summarize epoch metrics"""
        summary = {}
        
        for key, values in metrics.items():
            if values:  # Only process non-empty lists
                if isinstance(values[0], (int, float)):
                    summary[f'avg_{key}'] = np.mean(values)
                    summary[f'std_{key}'] = np.std(values)
                    summary[f'min_{key}'] = np.min(values)
                    summary[f'max_{key}'] = np.max(values)
                    
        return summary
    
    def run_tolerance_sweep(self):
        """Test different tolerance values to debug convergence"""
        print(f"\n🔬 TOLERANCE SWEEP EXPERIMENT")
        
        tolerances = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        results = {}
        
        for tol in tolerances:
            print(f"\n   Testing tolerance: {tol}")
            
            # Update model tolerance
            self.model.shooting_solver.tolerance = tol
            
            # Run a few mini-batches
            batch = next(iter(self.dataloader))
            c_sources, c_targets, wavelengths, _ = batch
            c_sources = c_sources.to(self.device)
            c_targets = c_targets.to(self.device)
            wavelengths = wavelengths.to(self.device)
            
            # Test shooting solver
            with torch.no_grad():
                bvp_results = self.model.shooting_solver.solve_batch(
                    c_sources[:64],  # Small batch for speed
                    c_targets[:64],
                    wavelengths[:64]
                )
                
            results[tol] = {
                'convergence_rate': bvp_results['convergence_rate'].item(),
                'mean_error': bvp_results['mean_error'].item(),
                'min_error': bvp_results['final_errors'].min().item(),
                'max_error': bvp_results['final_errors'].max().item()
            }
            
            print(f"     Convergence: {results[tol]['convergence_rate']:.1%}")
            print(f"     Error range: [{results[tol]['min_error']:.6f}, {results[tol]['max_error']:.6f}]")
            
        return results
    
    def run_smoothness_debug(self):
        """Debug smoothness loss computation step by step"""
        print(f"\n🔬 SMOOTHNESS LOSS DEBUG")
        
        # Get a small batch
        batch = next(iter(self.dataloader))
        c_sources, c_targets, wavelengths, _ = batch
        c_batch = c_sources[:8].to(self.device)  # Very small for debugging
        wl_batch = wavelengths[:8].to(self.device)
        
        print(f"   Testing with {len(c_batch)} samples")
        
        # Step 1: Check basic metric computation
        print(f"\n   Step 1: Basic metric computation")
        inputs = torch.stack([c_batch, wl_batch], dim=1)
        try:
            g_values = self.model.metric_network(inputs)
            print(f"     ✅ Metric values: [{g_values.min():.3f}, {g_values.max():.3f}]")
        except Exception as e:
            print(f"     ❌ Metric computation failed: {e}")
            return
            
        # Step 2: Check derivative computation
        print(f"\n   Step 2: Derivative computation")
        try:
            derivatives = self.model.metric_network.compute_derivatives(
                c_batch, wl_batch, create_graph=True
            )
            print(f"     ✅ First derivatives: [{derivatives['dg_dc'].min():.6f}, {derivatives['dg_dc'].max():.6f}]")
            
            if 'd2g_dc2' in derivatives:
                print(f"     ✅ Second derivatives: [{derivatives['d2g_dc2'].min():.6f}, {derivatives['d2g_dc2'].max():.6f}]")
            else:
                print(f"     ❌ Second derivatives not computed")
                
        except Exception as e:
            print(f"     ❌ Derivative computation failed: {e}")
            return
            
        # Step 3: Full smoothness loss
        print(f"\n   Step 3: Full smoothness loss computation")
        try:
            smoothness = self.model.metric_network.get_smoothness_loss(c_batch, wl_batch)
            print(f"     ✅ Smoothness loss: {smoothness.item():.6f}")
            print(f"     ✅ Requires grad: {smoothness.requires_grad}")
        except Exception as e:
            print(f"     ❌ Smoothness loss failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='Debug Geodesic NODE Training')
    parser.add_argument('--mode', choices=['baseline', 'tolerance_sweep', 'smoothness_debug', 'full_debug'], 
                       default='baseline', help='Debug mode to run')
    parser.add_argument('--data_path', type=str, default=None, help='Path to data CSV')
    parser.add_argument('--epochs', type=int, default=3, help='Number of debug epochs')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU usage (avoid MPS issues)')
    
    args = parser.parse_args()
    
    # Create debug session
    debug_session = DebugTrainingSession(mode=args.mode, data_path=args.data_path, force_cpu=args.force_cpu)
    
    if args.mode == 'baseline':
        # Run baseline debug epochs
        for epoch in range(args.epochs):
            print(f"\n{'='*60}")
            print(f"DEBUG EPOCH {epoch + 1}/{args.epochs}")
            print(f"{'='*60}")
            
            epoch_summary = debug_session.run_debug_epoch()
            
            print(f"\n📊 EPOCH {epoch + 1} SUMMARY:")
            for key, value in epoch_summary.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.6f}")
                    
    elif args.mode == 'tolerance_sweep':
        results = debug_session.run_tolerance_sweep()
        print(f"\n📊 TOLERANCE SWEEP RESULTS:")
        for tol, result in results.items():
            print(f"   {tol}: {result['convergence_rate']:.1%} convergence, {result['mean_error']:.6f} error")
            
    elif args.mode == 'smoothness_debug':
        debug_session.run_smoothness_debug()
        
    elif args.mode == 'full_debug':
        print(f"\n🔬 RUNNING FULL DEBUG SUITE")
        debug_session.run_smoothness_debug()
        debug_session.run_tolerance_sweep()
        
        for epoch in range(2):  # Just 2 epochs for full debug
            print(f"\n{'='*60}")
            print(f"DEBUG EPOCH {epoch + 1}/2")
            print(f"{'='*60}")
            debug_session.run_debug_epoch()
    
    print(f"\n✅ Debug session complete!")
    print(f"💾 Memory info: {debug_session.device_manager.get_memory_info()}")


if __name__ == "__main__":
    main()