#!/usr/bin/env python3
"""
Fast training script with optimizations for reasonable training time
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
import time

from data_loader import create_data_loaders, SpectralDataset
from geodesic_model import GeodesicSpectralModel
from train import Trainer


def create_fast_model():
    """Create model with faster settings"""
    return GeodesicSpectralModel(
        n_trajectory_points=7,  # Fewer points (was 11)
        shooting_tolerance=1e-3,  # Looser tolerance (was 1e-4)
        shooting_max_iter=20  # Fewer iterations (was 50)
    )


class FastTrainer(Trainer):
    """Optimized trainer with sampling and caching"""
    
    def __init__(self, model, sample_ratio=0.1, **kwargs):
        """
        Initialize fast trainer
        
        Args:
            model: Model to train
            sample_ratio: Fraction of data to use per epoch (0.1 = 10%)
            **kwargs: Other arguments for base Trainer
        """
        super().__init__(model, **kwargs)
        self.sample_ratio = sample_ratio
        
    def train_epoch_sampled(self, dataloader):
        """Train on a subset of data for speed"""
        self.model.train()
        epoch_losses = {
            'total': [],
            'reconstruction': [],
            'smoothness': [],
            'bounds': [],
            'path_efficiency': []
        }
        geodesic_successes = []
        
        # Sample subset of batches
        total_batches = len(dataloader)
        n_sample_batches = max(1, int(total_batches * self.sample_ratio))
        sampled_indices = np.random.choice(total_batches, n_sample_batches, replace=False)
        
        progress_bar = tqdm(enumerate(dataloader), 
                           total=n_sample_batches, 
                           desc="Training (sampled)", 
                           leave=False)
        
        batch_count = 0
        for batch_idx, batch in progress_bar:
            if batch_idx not in sampled_indices:
                continue
                
            batch_count += 1
            if batch_count > n_sample_batches:
                break
            
            # Move batch to device
            c_source = batch['c_source'].to(self.device)
            c_target = batch['c_target'].to(self.device)
            wavelength = batch['wavelength'].to(self.device)
            absorbance_target = batch['absorbance_target'].to(self.device)
            
            # Forward pass
            outputs = self.model(c_source, c_target, wavelength)
            
            # Track geodesic success rate
            geodesic_successes.extend(outputs['success_mask'].cpu().numpy())
            
            # Compute losses (fewer regularization samples for speed)
            c_samples = torch.randn(5, device=self.device)  # Was 20
            wavelength_samples = torch.randn(5, device=self.device)
            
            losses = self.model.compute_loss(
                outputs, absorbance_target, c_samples, wavelength_samples
            )
            
            # Combine losses
            total_loss = self.model.combine_losses(losses, self.loss_weights)
            
            # Backward pass
            self.optimizer_metric.zero_grad()
            self.optimizer_decoder.zero_grad()
            
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update parameters
            self.optimizer_metric.step()
            self.optimizer_decoder.step()
            
            # Record losses
            epoch_losses['total'].append(total_loss.item())
            for key in ['reconstruction', 'smoothness', 'bounds', 'path_efficiency']:
                if key in losses:
                    epoch_losses[key].append(losses[key].item())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'geo_success': f"{outputs['success_mask'].float().mean():.1%}"
            })
        
        # Compute epoch averages
        avg_losses = {key: np.mean(vals) if vals else 0.0 
                     for key, vals in epoch_losses.items()}
        avg_losses['geodesic_success_rate'] = np.mean(geodesic_successes) if geodesic_successes else 0.0
        
        return avg_losses
    
    def train_fast(self, n_epochs=50, exclude_concentration_idx=None):
        """Fast training with reduced epochs and sampling"""
        print(f"Starting FAST training for {n_epochs} epochs...")
        print(f"Sampling ratio: {self.sample_ratio:.1%} of data per epoch")
        print(f"Batch size: {self.batch_size}")
        
        # Create data loader
        dataloader, dataset = create_data_loaders(
            batch_size=self.batch_size,
            exclude_concentration_idx=exclude_concentration_idx
        )
        
        print(f"Total samples: {len(dataset)}")
        print(f"Samples per epoch: ~{int(len(dataset) * self.sample_ratio)}")
        
        # Training loop
        best_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(1, n_epochs + 1):
            epoch_start = time.time()
            print(f"\nEpoch {epoch}/{n_epochs}")
            
            # Train epoch with sampling
            avg_losses = self.train_epoch_sampled(dataloader)
            
            # Update learning rate
            self.scheduler_metric.step()
            self.scheduler_decoder.step()
            
            # Record history
            self.history['epoch'].append(epoch)
            self.history['total_loss'].append(avg_losses['total'])
            self.history['reconstruction_loss'].append(avg_losses.get('reconstruction', 0))
            self.history['geodesic_success_rate'].append(avg_losses['geodesic_success_rate'])
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            print(f"  Total Loss: {avg_losses['total']:.6f}")
            print(f"  Reconstruction: {avg_losses.get('reconstruction', 0):.6f}")
            print(f"  Geodesic Success: {avg_losses['geodesic_success_rate']:.1%}")
            print(f"  Epoch time: {epoch_time:.1f}s, Total time: {total_time/60:.1f}min")
            
            # Estimate remaining time
            avg_epoch_time = total_time / epoch
            remaining_epochs = n_epochs - epoch
            eta_seconds = avg_epoch_time * remaining_epochs
            print(f"  ETA: {eta_seconds/60:.1f} minutes")
            
            # Save best model
            if avg_losses['total'] < best_loss:
                best_loss = avg_losses['total']
                self.save_checkpoint(epoch, avg_losses, is_best=True)
                print(f"  → New best model saved (loss: {best_loss:.6f})")
            
            # Regular checkpoint every 10 epochs
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, avg_losses, is_best=False)
                print(f"  → Checkpoint saved")
        
        total_training_time = time.time() - start_time
        print(f"\nTraining completed in {total_training_time/60:.1f} minutes!")
        self.print_summary()


def main():
    """Main fast training script"""
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("="*60)
    print("FAST GEODESIC MODEL TRAINING")
    print("="*60)
    
    # Create optimized model
    model = create_fast_model()
    
    # Create fast trainer
    trainer = FastTrainer(
        model=model,
        lr_metric=5e-4,
        lr_decoder=1e-3,
        batch_size=64,  # Larger batch for efficiency
        n_epochs=50,  # Will be overridden
        checkpoint_dir="checkpoints_fast",
        sample_ratio=0.1  # Use 10% of data per epoch
    )
    
    # Train with different configurations based on user choice
    print("\nTraining configurations available:")
    print("1. Quick test (10 epochs, 10% sampling) - ~5 minutes")
    print("2. Standard (50 epochs, 10% sampling) - ~25 minutes")
    print("3. Extended (100 epochs, 20% sampling) - ~90 minutes")
    
    # For automatic execution, using option 1 (quick test)
    print("\nRunning quick test configuration...")
    trainer.sample_ratio = 0.1
    trainer.train_fast(n_epochs=10, exclude_concentration_idx=None)
    
    print("\n" + "="*60)
    print("Training complete! For longer training, modify the script.")
    print("="*60)


if __name__ == "__main__":
    main()