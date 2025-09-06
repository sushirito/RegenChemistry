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
from geodesic_a100.core.christoffel_computer import ChristoffelComputer
from geodesic_a100.core.geodesic_integrator import GeodesicIntegrator
from geodesic_a100.core.shooting_solver import ShootingSolver
from geodesic_a100.data.data_loader import SpectralDataset


class DebugTrainingSession:
    """Debug training session with enhanced diagnostics"""
    
    def __init__(self, mode: str = "baseline", data_path: str = None):
        self.mode = mode
        self.data_path = data_path or "data/preprocessed_arsenic_data.csv"
        
        # Setup device
        self.device_manager = M3DeviceManager(force_cpu=False, prefer_mps=True)
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
        batch_size = self.config['batch_size']
        self.dataloader = self.dataset.get_dataloader(batch_size=batch_size, shuffle=True)
        
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
            target_absorbance = target_absorbance.to(self.device)\n            \n            # DETAILED DIAGNOSTICS FOR FIRST BATCH\n            if batch_idx == 0:\n                self._print_batch_diagnostics(c_sources, c_targets, wavelengths, target_absorbance)\n            \n            # Zero gradients\n            self.optimizer_metric.zero_grad()\n            self.optimizer_flow.zero_grad()\n            \n            try:\n                # Forward pass with diagnostic wrapper\n                output = self._debug_forward_pass(c_sources, c_targets, wavelengths, batch_idx)\n                \n                # Loss computation with diagnostics\n                loss_dict = self._debug_loss_computation(\n                    output, target_absorbance, c_sources, wavelengths, batch_idx\n                )\n                \n                # Backward pass\n                loss = loss_dict['total']\n                loss.backward()\n                \n                # Gradient diagnostics\n                if batch_idx == 0:\n                    self._print_gradient_diagnostics()\n                \n                # Optimizer step\n                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)\n                self.optimizer_metric.step()\n                self.optimizer_flow.step()\n                \n                # Record metrics\n                epoch_metrics['total_loss'].append(loss.item())\n                epoch_metrics['reconstruction_loss'].append(loss_dict['reconstruction'].item())\n                epoch_metrics['smoothness_loss'].append(loss_dict['smoothness'].item())\n                epoch_metrics['bounds_loss'].append(loss_dict['bounds'].item())\n                epoch_metrics['path_length_loss'].append(loss_dict['path_length'].item())\n                epoch_metrics['convergence_rates'].append(output.get('convergence_rate', 0).item())\n                \n                # Memory usage\n                memory_info = self.device_manager.get_memory_info()\n                if 'allocated_gb' in memory_info:\n                    epoch_metrics['memory_usage'].append(memory_info['allocated_gb'])\n                \n                batch_time = time.time() - batch_start\n                epoch_metrics['batch_times'].append(batch_time)\n                \n            except Exception as e:\n                print(f\"   ❌ Error in batch {batch_idx}: {e}\")\n                if batch_idx == 0:  # Critical error in first batch\n                    raise e\n                continue  # Skip this batch\n                \n        return self._summarize_epoch_metrics(epoch_metrics)\n    \n    def _print_batch_diagnostics(self, c_sources, c_targets, wavelengths, target_absorbance):\n        """Print detailed diagnostics for first batch\"\"\"\n        print(f\"\\n📊 BATCH DIAGNOSTICS (First Batch):\")\n        print(f\"   Batch size: {c_sources.shape[0]}\")\n        print(f\"   c_sources range: [{c_sources.min():.3f}, {c_sources.max():.3f}]\")\n        print(f\"   c_targets range: [{c_targets.min():.3f}, {c_targets.max():.3f}]\")\n        print(f\"   |c_diff| range: [{(c_targets - c_sources).abs().min():.3f}, {(c_targets - c_sources).abs().max():.3f}]\")\n        print(f\"   wavelengths range: [{wavelengths.min():.3f}, {wavelengths.max():.3f}]\")\n        print(f\"   target_absorbance range: [{target_absorbance.min():.3f}, {target_absorbance.max():.3f}]\")\n        \n    def _debug_forward_pass(self, c_sources, c_targets, wavelengths, batch_idx):\n        \"\"\"Forward pass with shooting solver diagnostics\"\"\"\n        if batch_idx == 0:\n            print(f\"\\n🎯 SHOOTING SOLVER DIAGNOSTICS:\")\n            \n        # Custom forward to capture shooting diagnostics\n        bvp_results = self.model.shooting_solver.solve_batch(\n            c_sources=c_sources,\n            c_targets=c_targets,\n            wavelengths=wavelengths,\n            n_trajectory_points=self.model.n_trajectory_points\n        )\n        \n        if batch_idx == 0:\n            print(f\"   Convergence rate: {bvp_results['convergence_rate']:.1%}\")\n            print(f\"   Error range: [{bvp_results['final_errors'].min():.6f}, {bvp_results['final_errors'].max():.6f}]\")\n            print(f\"   Shooting tolerance: {self.model.shooting_solver.tolerance}\")\n            print(f\"   Converged geodesics: {bvp_results['convergence_mask'].sum().item()}/{len(c_sources)}\")\n            \n        return {\n            'absorbance': bvp_results['final_absorbance'],\n            'trajectories': bvp_results['trajectories'],\n            'initial_velocities': bvp_results['initial_velocities'],\n            'convergence_mask': bvp_results['convergence_mask'],\n            'convergence_rate': bvp_results['convergence_rate']\n        }\n    \n    def _debug_loss_computation(self, output, targets, c_batch, wavelength_batch, batch_idx):\n        \"\"\"Loss computation with smoothness diagnostics\"\"\"\n        # Reconstruction loss\n        reconstruction_loss = nn.functional.mse_loss(output['absorbance'], targets)\n        \n        if batch_idx == 0:\n            print(f\"\\n📉 LOSS DIAGNOSTICS:\")\n            print(f\"   Reconstruction loss: {reconstruction_loss.item():.6f}\")\n            \n        # DEBUG: Smoothness loss with step-by-step tracking\n        try:\n            print(f\"   Computing smoothness loss for {len(c_batch)} samples...\")\n            smoothness_loss = self.model.metric_network.get_smoothness_loss(c_batch, wavelength_batch)\n            if batch_idx == 0:\n                print(f\"   Smoothness loss: {smoothness_loss.item():.6f}\")\n        except Exception as e:\n            print(f\"   ❌ Smoothness loss failed: {e}\")\n            smoothness_loss = torch.tensor(0.0, device=self.device, requires_grad=True)\n            \n        # Bounds loss\n        bounds_loss = self.model.metric_network.get_bounds_loss(c_batch, wavelength_batch)\n        \n        # Path length loss\n        if 'trajectories' in output:\n            velocities = output['trajectories'][:, :, 1]\n            path_lengths = velocities.abs().mean(dim=0).mean()\n        else:\n            path_lengths = torch.tensor(0.0, device=self.device)\n            \n        if batch_idx == 0:\n            print(f\"   Bounds loss: {bounds_loss.item():.6f}\")\n            print(f\"   Path length loss: {path_lengths.item():.6f}\")\n            \n        # Total loss\n        total_loss = (\n            reconstruction_loss +\n            0.01 * smoothness_loss +\n            0.001 * bounds_loss +\n            0.001 * path_lengths\n        )\n        \n        return {\n            'total': total_loss,\n            'reconstruction': reconstruction_loss,\n            'smoothness': smoothness_loss,\n            'bounds': bounds_loss,\n            'path_length': path_lengths\n        }\n    \n    def _print_gradient_diagnostics(self):\n        \"\"\"Print gradient flow diagnostics\"\"\"\n        print(f\"\\n🔄 GRADIENT DIAGNOSTICS:\")\n        \n        total_norm = 0\n        param_count = 0\n        \n        for name, param in self.model.named_parameters():\n            if param.grad is not None:\n                param_norm = param.grad.data.norm(2)\n                total_norm += param_norm.item() ** 2\n                param_count += 1\n                if 'metric' in name:\n                    print(f\"   {name}: {param_norm:.6f}\")\n                    \n        total_norm = total_norm ** (1. / 2)\n        print(f\"   Total gradient norm: {total_norm:.6f}\")\n        print(f\"   Parameters with gradients: {param_count}\")\n    \n    def _summarize_epoch_metrics(self, metrics) -> dict:\n        \"\"\"Summarize epoch metrics\"\"\"\n        summary = {}\n        \n        for key, values in metrics.items():\n            if values:  # Only process non-empty lists\n                if isinstance(values[0], (int, float)):\n                    summary[f'avg_{key}'] = np.mean(values)\n                    summary[f'std_{key}'] = np.std(values)\n                    summary[f'min_{key}'] = np.min(values)\n                    summary[f'max_{key}'] = np.max(values)\n                    \n        return summary\n    \n    def run_tolerance_sweep(self):\n        \"\"\"Test different tolerance values to debug convergence\"\"\"\n        print(f\"\\n🔬 TOLERANCE SWEEP EXPERIMENT\")\n        \n        tolerances = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]\n        results = {}\n        \n        for tol in tolerances:\n            print(f\"\\n   Testing tolerance: {tol}\")\n            \n            # Update model tolerance\n            self.model.shooting_solver.tolerance = tol\n            \n            # Run a few mini-batches\n            batch = next(iter(self.dataloader))\n            c_sources, c_targets, wavelengths, _ = batch\n            c_sources = c_sources.to(self.device)\n            c_targets = c_targets.to(self.device)\n            wavelengths = wavelengths.to(self.device)\n            \n            # Test shooting solver\n            with torch.no_grad():\n                bvp_results = self.model.shooting_solver.solve_batch(\n                    c_sources[:64],  # Small batch for speed\n                    c_targets[:64],\n                    wavelengths[:64]\n                )\n                \n            results[tol] = {\n                'convergence_rate': bvp_results['convergence_rate'].item(),\n                'mean_error': bvp_results['mean_error'].item(),\n                'min_error': bvp_results['final_errors'].min().item(),\n                'max_error': bvp_results['final_errors'].max().item()\n            }\n            \n            print(f\"     Convergence: {results[tol]['convergence_rate']:.1%}\")\n            print(f\"     Error range: [{results[tol]['min_error']:.6f}, {results[tol]['max_error']:.6f}]\")\n            \n        return results\n    \n    def run_smoothness_debug(self):\n        \"\"\"Debug smoothness loss computation step by step\"\"\"\n        print(f\"\\n🔬 SMOOTHNESS LOSS DEBUG\")\n        \n        # Get a small batch\n        batch = next(iter(self.dataloader))\n        c_sources, c_targets, wavelengths, _ = batch\n        c_batch = c_sources[:8].to(self.device)  # Very small for debugging\n        wl_batch = wavelengths[:8].to(self.device)\n        \n        print(f\"   Testing with {len(c_batch)} samples\")\n        \n        # Step 1: Check basic metric computation\n        print(f\"\\n   Step 1: Basic metric computation\")\n        inputs = torch.stack([c_batch, wl_batch], dim=1)\n        try:\n            g_values = self.model.metric_network(inputs)\n            print(f\"     ✅ Metric values: [{g_values.min():.3f}, {g_values.max():.3f}]\")\n        except Exception as e:\n            print(f\"     ❌ Metric computation failed: {e}\")\n            return\n            \n        # Step 2: Check derivative computation\n        print(f\"\\n   Step 2: Derivative computation\")\n        try:\n            derivatives = self.model.metric_network.compute_derivatives(\n                c_batch, wl_batch, create_graph=True\n            )\n            print(f\"     ✅ First derivatives: [{derivatives['dg_dc'].min():.6f}, {derivatives['dg_dc'].max():.6f}]\")\n            \n            if 'd2g_dc2' in derivatives:\n                print(f\"     ✅ Second derivatives: [{derivatives['d2g_dc2'].min():.6f}, {derivatives['d2g_dc2'].max():.6f}]\")\n            else:\n                print(f\"     ❌ Second derivatives not computed\")\n                \n        except Exception as e:\n            print(f\"     ❌ Derivative computation failed: {e}\")\n            return\n            \n        # Step 3: Full smoothness loss\n        print(f\"\\n   Step 3: Full smoothness loss computation\")\n        try:\n            smoothness = self.model.metric_network.get_smoothness_loss(c_batch, wl_batch)\n            print(f\"     ✅ Smoothness loss: {smoothness.item():.6f}\")\n            print(f\"     ✅ Requires grad: {smoothness.requires_grad}\")\n        except Exception as e:\n            print(f\"     ❌ Smoothness loss failed: {e}\")\n\n\ndef main():\n    parser = argparse.ArgumentParser(description='Debug Geodesic NODE Training')\n    parser.add_argument('--mode', choices=['baseline', 'tolerance_sweep', 'smoothness_debug', 'full_debug'], \n                       default='baseline', help='Debug mode to run')\n    parser.add_argument('--data_path', type=str, default=None, help='Path to data CSV')\n    parser.add_argument('--epochs', type=int, default=3, help='Number of debug epochs')\n    \n    args = parser.parse_args()\n    \n    # Create debug session\n    debug_session = DebugTrainingSession(mode=args.mode, data_path=args.data_path)\n    \n    if args.mode == 'baseline':\n        # Run baseline debug epochs\n        for epoch in range(args.epochs):\n            print(f\"\\n{'='*60}\")\n            print(f\"DEBUG EPOCH {epoch + 1}/{args.epochs}\")\n            print(f\"{'='*60}\")\n            \n            epoch_summary = debug_session.run_debug_epoch()\n            \n            print(f\"\\n📊 EPOCH {epoch + 1} SUMMARY:\")\n            for key, value in epoch_summary.items():\n                if isinstance(value, float):\n                    print(f\"   {key}: {value:.6f}\")\n                    \n    elif args.mode == 'tolerance_sweep':\n        results = debug_session.run_tolerance_sweep()\n        print(f\"\\n📊 TOLERANCE SWEEP RESULTS:\")\n        for tol, result in results.items():\n            print(f\"   {tol}: {result['convergence_rate']:.1%} convergence, {result['mean_error']:.6f} error\")\n            \n    elif args.mode == 'smoothness_debug':\n        debug_session.run_smoothness_debug()\n        \n    elif args.mode == 'full_debug':\n        print(f\"\\n🔬 RUNNING FULL DEBUG SUITE\")\n        debug_session.run_smoothness_debug()\n        debug_session.run_tolerance_sweep()\n        \n        for epoch in range(2):  # Just 2 epochs for full debug\n            print(f\"\\n{'='*60}\")\n            print(f\"DEBUG EPOCH {epoch + 1}/2\")\n            print(f\"{'='*60}\")\n            debug_session.run_debug_epoch()\n    \n    print(f\"\\n✅ Debug session complete!\")\n    print(f\"💾 Memory info: {debug_session.device_manager.get_memory_info()}\")\n\n\nif __name__ == \"__main__\":\n    main()