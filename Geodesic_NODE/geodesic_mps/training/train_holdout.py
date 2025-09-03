#!/usr/bin/env python3
"""
Quick training for leave-one-out validation
Trains a geodesic model excluding one specific concentration
"""

import torch
import torch.nn as nn
import time
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.geodesic_model import ParallelGeodesicModel
from training.data_generator import SpectralDataGenerator
from typing import Optional, Dict


def train_holdout_model(exclude_concentration_idx: int, 
                       device: torch.device,
                       n_epochs: int = 10,
                       save_path: Optional[str] = None,
                       verbose: bool = True) -> ParallelGeodesicModel:
    """
    Train a geodesic model excluding one concentration for leave-one-out validation
    
    Args:
        exclude_concentration_idx: Index of concentration to exclude (0-5)
        device: Device to train on
        n_epochs: Number of training epochs
        save_path: Path to save the trained model
        verbose: Print progress
        
    Returns:
        Trained model
    """
    if verbose:
        print(f"Training geodesic model excluding concentration index {exclude_concentration_idx}")
        
    # Create data generator excluding the holdout concentration
    data_gen = SpectralDataGenerator(
        device=device, 
        use_real_data=True,
        exclude_concentration_idx=exclude_concentration_idx
    )
    
    if verbose:
        all_concs = [0, 10, 20, 30, 40, 60]
        excluded_conc = all_concs[exclude_concentration_idx]
        remaining_concs = [c for i, c in enumerate(all_concs) if i != exclude_concentration_idx]
        print(f"  Excluded concentration: {excluded_conc} ppb")
        print(f"  Training on: {remaining_concs} ppb")
    
    # Create model
    model = ParallelGeodesicModel(
        device=device,
        metric_hidden_dim=8,
        spectral_hidden_dim=16,
        n_trajectory_points=5,
        shooting_max_iter=3
    )
    
    # Training configuration
    config = {
        'mega_batch_size': 64,      # Smaller for faster training
        'micro_batch_size': 16,
        'metric_lr': 0.0005,
        'spectral_lr': 0.001,
        'n_epochs': n_epochs
    }
    
    # Create optimizers
    metric_optimizer = torch.optim.Adam(
        list(model.metric_network.parameters()) + 
        list(model.christoffel_computer.parameters()) if hasattr(model, 'christoffel_computer') else [],
        lr=config['metric_lr']
    )
    
    spectral_optimizer = torch.optim.Adam(
        list(model.spectral_flow_network.parameters()) +
        list(model.coupled_ode.parameters()) + 
        list(model.shooting_solver.parameters()),
        lr=config['spectral_lr']
    )
    
    # Training loop
    model.train()
    losses = []
    
    start_time = time.time()
    
    for epoch in range(config['n_epochs']):
        epoch_start = time.time()
        epoch_losses = []
        
        # Sample training data
        n_samples = min(500, len(data_gen.transitions['pairs']) * 10)  # Quick training
        batch = data_gen.get_batch(n_samples)
        
        # Process in micro-batches
        total_samples = batch['source_conc_norm'].shape[0]
        n_micro_batches = (total_samples + config['micro_batch_size'] - 1) // config['micro_batch_size']
        
        for micro_batch_idx in range(n_micro_batches):
            start_idx = micro_batch_idx * config['micro_batch_size']
            end_idx = min(start_idx + config['micro_batch_size'], total_samples)
            
            # Extract micro-batch
            micro_batch = {
                'source_conc_norm': batch['source_conc_norm'][start_idx:end_idx],
                'target_conc_norm': batch['target_conc_norm'][start_idx:end_idx], 
                'wavelength_norm': batch['wavelength_norm'][start_idx:end_idx],
                'source_absorbance': batch['source_absorbance'][start_idx:end_idx],
                'target_absorbance': batch['target_absorbance'][start_idx:end_idx]
            }
            
            # Forward pass
            try:
                result = model.forward_batch(micro_batch)
                predicted = result['absorbance']
                target = micro_batch['target_absorbance']
                
                # Compute loss
                loss = nn.MSELoss()(predicted, target)
                
                # Backward pass
                metric_optimizer.zero_grad()
                spectral_optimizer.zero_grad()
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimization step
                metric_optimizer.step()
                spectral_optimizer.step()
                
                epoch_losses.append(loss.item())
                
            except Exception as e:
                if verbose:
                    print(f"    Warning: Batch failed with {e}")
                continue
        
        # Record epoch results
        if epoch_losses:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(avg_loss)
            
            if verbose:
                epoch_time = time.time() - epoch_start
                print(f"  Epoch {epoch+1:2d}/{config['n_epochs']:2d}: "
                      f"Loss={avg_loss:.6f}, Time={epoch_time:.1f}s")
        else:
            if verbose:
                print(f"  Epoch {epoch+1:2d}/{config['n_epochs']:2d}: No successful batches")
    
    total_time = time.time() - start_time
    
    if verbose:
        print(f"Training complete! Total time: {total_time:.1f}s")
        if losses:
            print(f"Final loss: {losses[-1]:.6f}")
    
    # Save model if requested
    if save_path:
        torch.save({
            'model_state': model.state_dict(),
            'config': config,
            'excluded_concentration_idx': exclude_concentration_idx,
            'training_losses': losses,
            'training_time': total_time
        }, save_path)
        if verbose:
            print(f"Model saved to {save_path}")
    
    model.eval()
    return model


def train_all_holdout_models(device: torch.device,
                           n_epochs: int = 10,
                           save_dir: str = "checkpoints/holdout_models",
                           verbose: bool = True) -> Dict[int, ParallelGeodesicModel]:
    """
    Train geodesic models for all 6 leave-one-out cases
    
    Returns:
        Dictionary mapping holdout index to trained model
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    models = {}
    concentrations = [0, 10, 20, 30, 40, 60]
    
    total_start = time.time()
    
    for holdout_idx in range(6):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training model {holdout_idx+1}/6: Excluding {concentrations[holdout_idx]} ppb")
            print(f"{'='*60}")
        
        save_path = f"{save_dir}/model_exclude_{holdout_idx}.pt"
        
        model = train_holdout_model(
            exclude_concentration_idx=holdout_idx,
            device=device,
            n_epochs=n_epochs,
            save_path=save_path,
            verbose=verbose
        )
        
        models[holdout_idx] = model
    
    total_time = time.time() - total_start
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"All models trained! Total time: {total_time:.1f}s")
        print(f"Average time per model: {total_time/6:.1f}s")
        print(f"{'='*60}")
    
    return models


if __name__ == "__main__":
    # Quick test
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train one model as test
    model = train_holdout_model(
        exclude_concentration_idx=5,  # Exclude 60 ppb
        device=device,
        n_epochs=5,
        verbose=True
    )
    
    print("✅ Holdout training test complete!")