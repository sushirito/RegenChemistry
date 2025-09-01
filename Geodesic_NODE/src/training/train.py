#!/usr/bin/env python3
"""
Training script for Geodesic Spectral Model
Implements the full training loop with all loss components
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
from typing import Dict, Optional

from src.data.data_loader import create_data_loaders, SpectralDataset
from src.models.geodesic_model import GeodesicSpectralModel


class Trainer:
    """
    Trainer for the Geodesic Spectral Model
    """
    
    def __init__(self,
                 model: GeodesicSpectralModel,
                 lr_metric: float = 5e-4,
                 lr_decoder: float = 1e-3,
                 batch_size: int = 32,
                 n_epochs: int = 500,
                 checkpoint_dir: str = "checkpoints",
                 device: str = "cpu"):
        """
        Initialize trainer
        
        Args:
            model: The geodesic model to train
            lr_metric: Learning rate for metric network
            lr_decoder: Learning rate for decoder network
            batch_size: Training batch size
            n_epochs: Number of training epochs
            checkpoint_dir: Directory for saving checkpoints
            device: Device to train on
        """
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.checkpoint_dir = checkpoint_dir
        
        # Create optimizers with different learning rates
        self.optimizer_metric = optim.Adam(
            model.metric_network.parameters(),
            lr=lr_metric
        )
        self.optimizer_decoder = optim.Adam(
            model.decoder.parameters(),
            lr=lr_decoder
        )
        
        # Learning rate schedulers (cosine annealing)
        self.scheduler_metric = CosineAnnealingLR(
            self.optimizer_metric, T_max=n_epochs
        )
        self.scheduler_decoder = CosineAnnealingLR(
            self.optimizer_decoder, T_max=n_epochs
        )
        
        # Loss weights (as specified in document)
        self.loss_weights = {
            'reconstruction': 1.0,
            'smoothness': 0.01,
            'bounds': 0.001,
            'path_efficiency': 0.001
        }
        
        # Training history
        self.history = {
            'epoch': [],
            'total_loss': [],
            'reconstruction_loss': [],
            'smoothness_loss': [],
            'bounds_loss': [],
            'path_loss': [],
            'geodesic_success_rate': []
        }
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            dataloader: Training data loader
        
        Returns:
            Dictionary of average losses for the epoch
        """
        self.model.train()
        epoch_losses = {
            'total': [],
            'reconstruction': [],
            'smoothness': [],
            'bounds': [],
            'path_efficiency': []
        }
        geodesic_successes = []
        
        progress_bar = tqdm(dataloader, desc="Training", leave=False)
        
        for batch in progress_bar:
            # Move batch to device
            c_source = batch['c_source'].to(self.device)
            c_target = batch['c_target'].to(self.device)
            wavelength = batch['wavelength'].to(self.device)
            absorbance_target = batch['absorbance_target'].to(self.device)
            
            # Forward pass
            outputs = self.model(c_source, c_target, wavelength)
            
            # Track geodesic success rate
            geodesic_successes.extend(outputs['success_mask'].cpu().numpy())
            
            # Compute losses
            # Sample points for regularization
            c_samples = torch.randn(20, device=self.device)
            wavelength_samples = torch.randn(20, device=self.device)
            
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
        avg_losses['geodesic_success_rate'] = np.mean(geodesic_successes)
        
        return avg_losses
    
    def train(self, 
              exclude_concentration_idx: Optional[int] = None,
              save_every: int = 50):
        """
        Full training loop
        
        Args:
            exclude_concentration_idx: Concentration to hold out for validation
            save_every: Save checkpoint every N epochs
        """
        print(f"Starting training for {self.n_epochs} epochs...")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rates - Metric: {self.optimizer_metric.param_groups[0]['lr']:.1e}, "
              f"Decoder: {self.optimizer_decoder.param_groups[0]['lr']:.1e}")
        
        # Create data loader
        dataloader, dataset = create_data_loaders(
            batch_size=self.batch_size,
            exclude_concentration_idx=exclude_concentration_idx
        )
        
        print(f"Training samples: {len(dataset)}")
        if exclude_concentration_idx is not None:
            print(f"Holding out concentration index {exclude_concentration_idx}")
        
        # Training loop
        best_loss = float('inf')
        
        for epoch in range(1, self.n_epochs + 1):
            print(f"\nEpoch {epoch}/{self.n_epochs}")
            
            # Train epoch
            avg_losses = self.train_epoch(dataloader)
            
            # Update learning rate
            self.scheduler_metric.step()
            self.scheduler_decoder.step()
            
            # Record history
            self.history['epoch'].append(epoch)
            self.history['total_loss'].append(avg_losses['total'])
            self.history['reconstruction_loss'].append(avg_losses.get('reconstruction', 0))
            self.history['smoothness_loss'].append(avg_losses.get('smoothness', 0))
            self.history['bounds_loss'].append(avg_losses.get('bounds', 0))
            self.history['path_loss'].append(avg_losses.get('path_efficiency', 0))
            self.history['geodesic_success_rate'].append(avg_losses['geodesic_success_rate'])
            
            # Print epoch summary
            print(f"  Total Loss: {avg_losses['total']:.6f}")
            print(f"  Reconstruction: {avg_losses.get('reconstruction', 0):.6f}")
            print(f"  Geodesic Success: {avg_losses['geodesic_success_rate']:.1%}")
            print(f"  LR - Metric: {self.scheduler_metric.get_last_lr()[0]:.1e}, "
                  f"Decoder: {self.scheduler_decoder.get_last_lr()[0]:.1e}")
            
            # Save best model
            if avg_losses['total'] < best_loss:
                best_loss = avg_losses['total']
                self.save_checkpoint(epoch, avg_losses, is_best=True)
                print(f"  → New best model saved (loss: {best_loss:.6f})")
            
            # Regular checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, avg_losses, is_best=False)
                print(f"  → Checkpoint saved")
        
        print("\nTraining completed!")
        self.print_summary()
    
    def save_checkpoint(self, epoch: int, losses: Dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_metric_state_dict': self.optimizer_metric.state_dict(),
            'optimizer_decoder_state_dict': self.optimizer_decoder.state_dict(),
            'scheduler_metric_state_dict': self.scheduler_metric.state_dict(),
            'scheduler_decoder_state_dict': self.scheduler_decoder.state_dict(),
            'losses': losses,
            'history': self.history,
            'loss_weights': self.loss_weights
        }
        
        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}_{timestamp}.pth')
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_metric.load_state_dict(checkpoint['optimizer_metric_state_dict'])
        self.optimizer_decoder.load_state_dict(checkpoint['optimizer_decoder_state_dict'])
        self.scheduler_metric.load_state_dict(checkpoint['scheduler_metric_state_dict'])
        self.scheduler_decoder.load_state_dict(checkpoint['scheduler_decoder_state_dict'])
        self.history = checkpoint['history']
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Losses at checkpoint: {checkpoint['losses']}")
    
    def print_summary(self):
        """Print training summary"""
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        
        # Final losses
        print(f"Final Total Loss: {self.history['total_loss'][-1]:.6f}")
        print(f"Final Reconstruction Loss: {self.history['reconstruction_loss'][-1]:.6f}")
        print(f"Final Geodesic Success Rate: {self.history['geodesic_success_rate'][-1]:.1%}")
        
        # Best performance
        best_epoch = np.argmin(self.history['total_loss']) + 1
        print(f"\nBest Epoch: {best_epoch}")
        print(f"Best Total Loss: {min(self.history['total_loss']):.6f}")
        
        # Model statistics
        stats = self.model.get_stats()
        print(f"\nModel Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


def main():
    """Main training script"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create model
    model = GeodesicSpectralModel(
        n_trajectory_points=11,
        shooting_tolerance=1e-4,
        shooting_max_iter=50
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        lr_metric=5e-4,
        lr_decoder=1e-3,
        batch_size=32,
        n_epochs=500,
        checkpoint_dir="checkpoints"
    )
    
    # Train model (no holdout for now, full training)
    trainer.train(exclude_concentration_idx=None, save_every=100)
    
    print("\nTraining script completed!")


if __name__ == "__main__":
    main()