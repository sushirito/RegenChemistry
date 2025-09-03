"""
M1 Mac optimized trainer for Geodesic-Coupled Spectral NODE
Handles multi-model leave-one-out training with MPS acceleration
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
import time
import os
from pathlib import Path

from geodesic_m1.models.geodesic_model import GeodesicNODE
from geodesic_m1.core.device_manager import M1DeviceManager
from geodesic_m1.training.data_loader import SpectralDataLoader, SpectralDataset
from geodesic_m1.training.mixed_precision import M1MixedPrecision
from geodesic_m1.training.validator import LeaveOneOutValidator


class M1Trainer:
    """M1 Mac optimized trainer for geodesic NODE models"""
    
    def __init__(self,
                 device: torch.device = None,
                 checkpoint_dir: str = "./checkpoints",
                 verbose: bool = True):
        """
        Initialize M1 trainer
        
        Args:
            device: Computation device (auto-detect M1 if None)
            checkpoint_dir: Directory for saving checkpoints
            verbose: Whether to print training progress
        """
        # Initialize device manager
        self.device_manager = M1DeviceManager()
        self.device = device or self.device_manager.device
        self.verbose = verbose
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize mixed precision
        self.mixed_precision = M1MixedPrecision(device=self.device)
        
        # Training state
        self.models = []
        self.optimizers = []
        self.training_history = {}
        self.best_losses = []
        
        if self.verbose:
            self.device_manager.print_system_info()
            self.mixed_precision.print_status()
            
    def create_model(self, config: Dict[str, Any], dataset: Optional[SpectralDataset] = None) -> GeodesicNODE:
        """
        Create and configure a geodesic NODE model
        
        Args:
            config: Model configuration dictionary
            dataset: Optional dataset for absorbance lookup
            
        Returns:
            Configured GeodesicNODE model
        """
        # Extract data for absorbance lookup if dataset provided
        concentrations = None
        wavelengths = None
        absorbance_matrix = None
        
        if dataset is not None:
            concentrations = dataset.concentration_values
            wavelengths = dataset.wavelengths
            absorbance_matrix = dataset.absorbance_data
        
        model = GeodesicNODE(
            metric_hidden_dims=config.get('metric_hidden_dims', [128, 256]),
            flow_hidden_dims=config.get('flow_hidden_dims', [64, 128]),
            n_trajectory_points=config.get('n_trajectory_points', 50),
            shooting_max_iter=config.get('shooting_max_iter', 10),
            shooting_tolerance=config.get('shooting_tolerance', 1e-4),
            shooting_learning_rate=config.get('shooting_learning_rate', 0.5),
            christoffel_grid_size=config.get('christoffel_grid_size', (2000, 601)),
            device=self.device,
            use_adjoint=config.get('use_adjoint', False),
            concentrations=concentrations,
            wavelengths=wavelengths,
            absorbance_matrix=absorbance_matrix
        )
        
        if self.verbose:
            model_info = model.get_model_info()
            print(f"🧠 Created model with {model_info['total_parameters']:,} parameters")
            
        return model
        
    def create_optimizer(self, model: GeodesicNODE, config: Dict[str, Any]) -> Dict[str, optim.Optimizer]:
        """
        Create optimizers with different learning rates for different components
        
        Args:
            model: GeodesicNODE model
            config: Optimizer configuration
            
        Returns:
            Dictionary of optimizers
        """
        # Separate parameters for different learning rates
        metric_params = list(model.metric_network.parameters())
        flow_params = list(model.spectral_flow_network.parameters())
        
        # Add absorbance lookup parameters if it exists
        if hasattr(model, 'absorbance_lookup') and model.absorbance_lookup is not None:
            lookup_params = list(model.absorbance_lookup.interpolation_network.parameters())
            flow_params.extend(lookup_params)
        
        # Create optimizers with different learning rates
        optimizers = {
            'metric': optim.Adam(
                metric_params,
                lr=config.get('metric_lr', 5e-4),
                weight_decay=config.get('weight_decay', 1e-5)
            ),
            'flow': optim.Adam(
                flow_params,
                lr=config.get('flow_lr', 1e-3),
                weight_decay=config.get('weight_decay', 1e-5)
            )
        }
        
        return optimizers
        
    def train_single_model(self,
                          model: GeodesicNODE,
                          dataloader: SpectralDataLoader,
                          optimizers: Dict[str, optim.Optimizer],
                          config: Dict[str, Any],
                          model_idx: int = 0) -> Dict[str, List[float]]:
        """
        Train a single geodesic NODE model
        
        Args:
            model: Model to train
            dataloader: Training data loader
            optimizers: Dictionary of optimizers
            config: Training configuration
            model_idx: Model index for logging
            
        Returns:
            Training history dictionary
        """
        epochs = config.get('epochs', 50)
        gradient_clip_norm = config.get('gradient_clip_norm', 1.0)
        log_interval = config.get('log_interval', 10)
        
        # Training history
        history = {
            'epoch': [],
            'total_loss': [],
            'reconstruction_loss': [],
            'smoothness_loss': [],
            'bounds_loss': [],
            'path_length_loss': [],
            'convergence_rate': [],
            'lr_metric': [],
            'lr_flow': [],
            'epoch_time': []
        }
        
        if self.verbose:
            print(f"\n🚀 Training Model {model_idx} (excluding concentration {model_idx})")
            print(f"   Epochs: {epochs}, Batch size: {dataloader.batch_size}")
            
        model.train()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            epoch_losses = []
            epoch_convergence_rates = []
            
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Zero gradients
                for optimizer in optimizers.values():
                    optimizer.zero_grad()
                    
                # Forward pass with mixed precision
                with self.mixed_precision.autocast():
                    # Extract batch data
                    c_sources = batch['c_source']
                    c_targets = batch['c_target']
                    wavelengths = batch['wavelength']
                    target_absorbances = batch['target_absorbance']
                    
                    # Forward pass through model
                    predictions = model.forward(c_sources, c_targets, wavelengths)
                    
                    # Compute loss
                    loss_components = model.compute_loss(
                        predictions, target_absorbances, c_sources, wavelengths
                    )
                    
                    total_loss = loss_components['total']
                    
                # Scale loss and backward pass
                scaled_loss = self.mixed_precision.scale_loss(total_loss)
                scaled_loss.backward()
                
                # Gradient clipping
                if gradient_clip_norm > 0:
                    for optimizer in optimizers.values():
                        if self.mixed_precision.enabled:
                            self.mixed_precision.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                
                # Optimizer steps
                all_steps_taken = True
                for optimizer in optimizers.values():
                    step_taken = self.mixed_precision.step_optimizer(optimizer)
                    all_steps_taken = all_steps_taken and step_taken
                    
                # Record metrics (only if step was taken)
                if all_steps_taken:
                    epoch_losses.append(total_loss.item())
                    epoch_convergence_rates.append(predictions['convergence_rate'].item())
                    
            # Epoch statistics
            if epoch_losses:
                mean_loss = sum(epoch_losses) / len(epoch_losses)
                mean_convergence = sum(epoch_convergence_rates) / len(epoch_convergence_rates)
                
                history['epoch'].append(epoch + 1)
                history['total_loss'].append(mean_loss)
                history['convergence_rate'].append(mean_convergence)
                
                # Track learning rates
                current_lr_metric = optimizers['metric'].param_groups[0]['lr']
                current_lr_flow = optimizers['flow'].param_groups[0]['lr']
                history['lr_metric'].append(current_lr_metric)
                history['lr_flow'].append(current_lr_flow)
                
                # Track epoch time
                epoch_time = time.time() - epoch_start_time
                history['epoch_time'].append(epoch_time)
                
                # Update learning rate schedulers if any
                # TODO: Add scheduler support
                
                # Save best model
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    self._save_checkpoint(model, optimizers, epoch, mean_loss, model_idx)
                    
                # Logging
                epoch_time = time.time() - epoch_start_time
                if self.verbose and (epoch + 1) % log_interval == 0:
                    print(f"   Epoch {epoch+1:3d}/{epochs}: "
                          f"Loss = {mean_loss:.6f}, "
                          f"Conv = {mean_convergence:.1%}, "
                          f"Time = {epoch_time:.1f}s")
                    
                    # Memory monitoring
                    if (epoch + 1) % (log_interval * 5) == 0:
                        self.device_manager.monitor_memory_usage(f"Epoch {epoch+1}")
                        
            # Clear cache periodically
            if (epoch + 1) % 10 == 0:
                self.device_manager.clear_cache()
                
        if self.verbose:
            print(f"✅ Model {model_idx} training completed (best loss: {best_loss:.6f})")
        
        # Save history to CSV
        self.save_history_to_csv(history, model_idx)
            
        return history
    
    def save_history_to_csv(self, history: Dict[str, List], model_idx: int):
        """Save training history to CSV file"""
        import pandas as pd
        import os
        
        # Filter out empty lists and ensure all have same length
        non_empty = {k: v for k, v in history.items() if len(v) > 0}
        
        if not non_empty:
            if self.verbose:
                print(f"   ⚠️  No history to save for model {model_idx}")
            return
        
        # Ensure all lists have the same length
        max_len = max(len(v) for v in non_empty.values())
        for key in non_empty:
            while len(non_empty[key]) < max_len:
                non_empty[key].append(non_empty[key][-1] if non_empty[key] else 0)
        
        # Create DataFrame from history
        df = pd.DataFrame(non_empty)
        
        # Add model information
        df['model_idx'] = model_idx
        df['excluded_concentration'] = model_idx  # Since we exclude by index
        
        # Save to CSV
        csv_path = os.path.join('outputs', 'training_logs', f'training_history_model_{model_idx}.csv')
        df.to_csv(csv_path, index=False)
        
        if self.verbose:
            print(f"   💾 History saved to {csv_path}")
        
    def train_all_models(self,
                        datasets: List[SpectralDataset],
                        config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train all 6 models for leave-one-out validation
        
        Args:
            datasets: List of 6 training datasets (each excluding one concentration)
            config: Training configuration
            
        Returns:
            Complete training results
        """
        if len(datasets) != 6:
            raise ValueError("Expected 6 datasets for leave-one-out training")
            
        if self.verbose:
            print("🎯 Starting leave-one-out training on M1 Mac...")
            print(f"   Training 6 models, each excluding one concentration")
            
        # Get M1 optimizations
        m1_opts = self.device_manager.optimize_for_training(config.get('batch_size', 1024))
        config.update(m1_opts)
        
        start_time = time.time()
        self.models = []
        self.optimizers = []
        self.training_history = {}
        self.best_losses = []
        
        # Train each model
        for model_idx in range(6):
            if self.verbose:
                print(f"\n📊 Training model {model_idx + 1}/6...")
                
            # Create model with data and optimizer
            model = self.create_model(config, datasets[model_idx])
            optimizers = self.create_optimizer(model, config)
            
            # Create data loader
            dataloader = SpectralDataLoader(
                datasets[model_idx],
                batch_size=config['recommended_batch_size'],
                shuffle=True,
                num_workers=config.get('num_workers', 0)
            )
            
            # Pre-compute Christoffel grid
            model.precompute_christoffel_grid()
            
            # Train model
            history = self.train_single_model(
                model, dataloader, optimizers, config, model_idx
            )
            
            # Store results
            self.models.append(model)
            self.optimizers.append(optimizers)
            self.training_history[f'model_{model_idx}'] = history
            self.best_losses.append(min(history['total_loss']) if history['total_loss'] else float('inf'))
            
        total_time = time.time() - start_time
        
        # Training summary
        mean_best_loss = sum(self.best_losses) / len(self.best_losses)
        worst_model_idx = self.best_losses.index(max(self.best_losses))
        
        results = {
            'models': self.models,
            'training_history': self.training_history,
            'best_losses': self.best_losses,
            'mean_best_loss': mean_best_loss,
            'worst_model_idx': worst_model_idx,
            'total_training_time_hours': total_time / 3600,
            'config': config
        }
        
        if self.verbose:
            print(f"\n🎉 Leave-one-out training completed!")
            print(f"   Total time: {total_time/3600:.2f} hours")
            print(f"   Mean best loss: {mean_best_loss:.6f}")
            print(f"   Worst model: {worst_model_idx} (loss: {self.best_losses[worst_model_idx]:.6f})")
            
        return results
        
    def validate_models(self,
                       validation_datasets: List[SpectralDataset],
                       config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run validation on all trained models
        
        Args:
            validation_datasets: List of validation datasets
            config: Validation configuration
            
        Returns:
            Validation results
        """
        if not self.models:
            raise RuntimeError("No trained models available. Run train_all_models first.")
            
        validator = LeaveOneOutValidator(device=self.device, verbose=self.verbose)
        
        # Load best checkpoints
        for i, model in enumerate(self.models):
            checkpoint_path = self.checkpoint_dir / f"best_model_{i}.pt"
            if checkpoint_path.exists():
                model.load_checkpoint(str(checkpoint_path))
                
        # Run validation
        validation_results = validator.validate_all_models(
            self.models,
            validation_datasets,
            batch_size=config.get('validation_batch_size', 512)
        )
        
        # Save results
        results_path = self.checkpoint_dir / "validation_results.pt"
        validator.save_results(str(results_path))
        
        return validation_results
        
    def _save_checkpoint(self,
                        model: GeodesicNODE,
                        optimizers: Dict[str, optim.Optimizer],
                        epoch: int,
                        loss: float,
                        model_idx: int):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"best_model_{model_idx}.pt"
        model.save_checkpoint(
            str(checkpoint_path),
            epoch=epoch,
            optimizers=optimizers,
            best_loss=loss
        )
        
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        if not self.models:
            return {'status': 'No models trained'}
            
        return {
            'n_models': len(self.models),
            'best_losses': self.best_losses,
            'mean_best_loss': sum(self.best_losses) / len(self.best_losses),
            'total_parameters': sum(m.get_model_info()['total_parameters'] for m in self.models),
            'device': str(self.device),
            'mixed_precision_stats': self.mixed_precision.get_stats(),
            'memory_stats': self.device_manager.get_memory_stats()
        }