#!/usr/bin/env python3
"""
Training Script for MPS-Optimized Geodesic NODE
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Tuple
import time
from pathlib import Path
import json

from ..models.geodesic_model import ParallelGeodesicModel
from ..training.data_generator import SpectralDataGenerator
from ..training.losses import GeodesicLoss
from ..configs.train_config import ExperimentConfig
from ..utils.mps_utils import MPSDeviceManager


class Trainer:
    """
    Trainer for Geodesic Spectral Model
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize trainer
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.device = config.get_device()
        
        # Device manager
        self.device_manager = MPSDeviceManager()
        
        # Setup model
        self.model = self._create_model()
        
        # Setup data
        self.data_generator = SpectralDataGenerator(
            self.device,
            use_real_data=True
        )
        
        # Setup loss
        self.loss_fn = GeodesicLoss(
            reconstruction_weight=config.training.reconstruction_weight,
            smoothness_weight=config.training.smoothness_weight,
            bounds_weight=config.training.bounds_weight,
            path_weight=config.training.path_weight
        )
        
        # Setup optimizers
        self.metric_optimizer, self.spectral_optimizer = self._create_optimizers()
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.epoch_times = []
    
    def _create_model(self) -> nn.Module:
        """Create and initialize model"""
        model = ParallelGeodesicModel(
            metric_hidden_dim=self.config.model.metric_hidden_dim,
            spectral_hidden_dim=self.config.model.spectral_hidden_dim,
            n_trajectory_points=self.config.model.n_trajectory_points,
            shooting_max_iter=self.config.model.shooting_max_iter,
            use_christoffel_cache=self.config.model.use_christoffel_cache,
            device=self.device
        )
        
        # Optimize for device
        model = self.device_manager.optimize_model_for_device(model)
        
        return model
    
    def _create_optimizers(self) -> Tuple[optim.Optimizer, optim.Optimizer]:
        """Create dual optimizers for metric and spectral networks"""
        metric_params = self.model.metric_network.parameters()
        spectral_params = self.model.spectral_flow_network.parameters()
        
        metric_optimizer = optim.Adam(
            metric_params,
            lr=self.config.training.metric_lr,
            weight_decay=self.config.training.weight_decay
        )
        
        spectral_optimizer = optim.Adam(
            spectral_params,
            lr=self.config.training.spectral_lr,
            weight_decay=self.config.training.weight_decay
        )
        
        return metric_optimizer, spectral_optimizer
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch
        
        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()
        epoch_losses = []
        
        # Calculate number of batches
        n_transitions = len(self.data_generator.transitions['pairs'])
        n_wavelengths = len(self.data_generator.wavelengths)
        total_samples = n_transitions * n_wavelengths
        
        # Apply sampling ratio
        samples_per_epoch = int(total_samples * self.config.training.data_sampling_ratio)
        n_batches = samples_per_epoch // self.config.training.mega_batch_size
        
        for batch_idx in range(n_batches):
            # Get batch
            batch = self.data_generator.get_batch(
                self.config.training.mega_batch_size,
                self.config.training.wavelength_subset
            )
            
            # Process mega-batch in micro-batches
            batch_loss = self._train_mega_batch(batch)
            epoch_losses.append(batch_loss)
            
            # Log progress
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{n_batches}, Loss: {batch_loss:.6f}")
        
        # Compute epoch metrics
        metrics = {
            'loss': sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0,
            'n_batches': n_batches,
            'samples_processed': samples_per_epoch
        }
        
        return metrics
    
    def _train_mega_batch(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Train on a mega-batch using gradient accumulation
        
        Args:
            batch: Batch data dictionary
        
        Returns:
            Average loss for the batch
        """
        # Zero gradients
        self.metric_optimizer.zero_grad()
        self.spectral_optimizer.zero_grad()
        
        total_loss = 0
        n_micro_batches = self.config.training.gradient_accumulation_steps
        micro_batch_size = self.config.training.micro_batch_size
        
        for i in range(n_micro_batches):
            # Get micro-batch slice
            start_idx = i * micro_batch_size
            end_idx = min((i + 1) * micro_batch_size, len(batch['source_conc']))
            
            if start_idx >= end_idx:
                break
            
            micro_batch = {
                key: value[start_idx:end_idx] if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            
            # Forward pass
            predictions = self.model.forward_batch(micro_batch)
            
            # Compute loss
            losses = self.loss_fn(
                predictions,
                micro_batch['target_absorbance'],
                self.model.metric_network
            )
            
            # Scale loss for gradient accumulation
            scaled_loss = losses['total'] / n_micro_batches
            
            # Backward pass
            scaled_loss.backward()
            
            total_loss += losses['total'].item()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.training.gradient_clip_norm
        )
        
        # Optimizer steps
        self.metric_optimizer.step()
        self.spectral_optimizer.step()
        
        return total_loss / n_micro_batches
    
    def validate(self) -> Dict[str, float]:
        """
        Validate model
        
        Returns:
            Validation metrics
        """
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            # Get validation set
            val_data = self.data_generator.get_validation_set(n_samples=100)
            
            # Forward pass
            predictions = self.model.forward_batch(val_data)
            
            # Compute loss
            losses = self.loss_fn(
                predictions,
                val_data['target_absorbance']
            )
            
            val_losses.append(losses['total'].item())
        
        metrics = {
            'val_loss': sum(val_losses) / len(val_losses) if val_losses else 0
        }
        
        return metrics
    
    def train(self) -> None:
        """
        Main training loop
        """
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        for epoch in range(self.config.training.n_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch + 1}/{self.config.training.n_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['loss'])
            
            # Validate
            if (epoch + 1) % self.config.training.validation_interval == 0:
                val_metrics = self.validate()
                self.val_losses.append(val_metrics['val_loss'])
                
                print(f"  Validation Loss: {val_metrics['val_loss']:.6f}")
                
                # Early stopping
                if val_metrics['val_loss'] < self.best_loss - self.config.training.min_delta:
                    self.best_loss = val_metrics['val_loss']
                    self.patience_counter = 0
                    self.save_checkpoint(is_best=True)
                else:
                    self.patience_counter += 1
                    
                if self.patience_counter >= self.config.training.patience:
                    print("Early stopping triggered")
                    break
            
            # Save checkpoint
            if (epoch + 1) % self.config.training.checkpoint_interval == 0:
                self.save_checkpoint()
            
            # Track time
            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)
            
            print(f"  Train Loss: {train_metrics['loss']:.6f}")
            print(f"  Epoch Time: {epoch_time:.2f}s")
            
            # Clear cache periodically
            if (epoch + 1) % 10 == 0:
                self.device_manager.empty_cache()
        
        print("\nTraining completed!")
        self.save_final_results()
    
    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state': self.model.state_dict(),
            'metric_optimizer': self.metric_optimizer.state_dict(),
            'spectral_optimizer': self.spectral_optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config.to_dict()
        }
        
        filename = 'best_model.pt' if is_best else f'checkpoint_epoch_{self.current_epoch}.pt'
        torch.save(checkpoint, checkpoint_dir / filename)
        
        print(f"  Checkpoint saved: {filename}")
    
    def save_final_results(self) -> None:
        """Save final training results"""
        results_dir = Path(self.config.log_dir)
        results_dir.mkdir(exist_ok=True)
        
        results = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_val_loss': self.val_losses[-1] if self.val_losses else None,
            'best_val_loss': self.best_loss,
            'total_epochs': self.current_epoch + 1,
            'average_epoch_time': sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0,
            'total_training_time': sum(self.epoch_times),
            'config': self.config.to_dict()
        }
        
        with open(results_dir / 'training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {results_dir / 'training_results.json'}")


def test_training():
    """Test training pipeline"""
    print("Testing Training Pipeline...")
    
    # Use quick test config
    from geodesic_mps.configs.train_config import get_quick_test_config
    config = get_quick_test_config()
    config.training.n_epochs = 2  # Very short for testing
    
    # Create trainer
    trainer = Trainer(config)
    
    # Test single epoch
    print("\nTesting single epoch...")
    metrics = trainer.train_epoch()
    print(f"Epoch metrics: {metrics}")
    
    # Test validation
    print("\nTesting validation...")
    val_metrics = trainer.validate()
    print(f"Validation metrics: {val_metrics}")
    
    print("\nTraining test passed!")


if __name__ == "__main__":
    test_training()