#!/usr/bin/env python3
"""
Ultra-fast demo to show the model works
Uses minimal data and iterations
"""

import torch
import numpy as np
import time
from geodesic_model import GeodesicSpectralModel
from data_loader import SpectralDataset

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

print("="*60)
print("GEODESIC MODEL - ULTRA FAST DEMO")
print("="*60)

# Create model with fastest possible settings
print("\n1. Creating model with optimized settings...")
model = GeodesicSpectralModel(
    n_trajectory_points=5,  # Minimum points
    shooting_tolerance=5e-3,  # Very loose tolerance
    shooting_max_iter=10  # Minimal iterations
)

# Setup training
dataset = SpectralDataset()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(f"   Model has {sum(p.numel() for p in model.parameters())} parameters")
print(f"   Dataset has {len(dataset)} samples")

# Train on just 20 samples
print("\n2. Training on 20 samples for demonstration...")
print("-"*40)

n_samples = 20
losses = []

start_time = time.time()

for i in range(n_samples):
    # Get sample
    sample = dataset[i]
    
    # Convert to tensors
    c_source = torch.tensor([sample['c_source']], dtype=torch.float32)
    c_target = torch.tensor([sample['c_target']], dtype=torch.float32)
    wavelength = torch.tensor([sample['wavelength']], dtype=torch.float32)
    target = torch.tensor([sample['absorbance_target']], dtype=torch.float32)
    
    # Forward pass
    outputs = model(c_source, c_target, wavelength)
    
    # Loss
    loss = torch.nn.functional.mse_loss(outputs['absorbance'], target)
    losses.append(loss.item())
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print every 5 samples
    if (i + 1) % 5 == 0:
        avg_loss = np.mean(losses[-5:])
        print(f"   Samples {i-3}-{i+1}: Avg Loss = {avg_loss:.6f}")

training_time = time.time() - start_time

print("-"*40)
print(f"\n3. Training Summary:")
print(f"   Total time: {training_time:.2f} seconds")
print(f"   Time per sample: {training_time/n_samples:.3f} seconds")
print(f"   Initial loss: {losses[0]:.6f}")
print(f"   Final loss: {losses[-1]:.6f}")
print(f"   Loss reduction: {(losses[0] - losses[-1])/losses[0]*100:.1f}%")

# Test prediction improvement
print(f"\n4. Testing prediction improvement...")
model.eval()
test_sample = dataset[100]  # Different sample

with torch.no_grad():
    # Before training (reinitialize for comparison)
    model_untrained = GeodesicSpectralModel(
        n_trajectory_points=5,
        shooting_tolerance=5e-3,
        shooting_max_iter=10
    )
    
    c_s = torch.tensor([test_sample['c_source']], dtype=torch.float32)
    c_t = torch.tensor([test_sample['c_target']], dtype=torch.float32)
    wl = torch.tensor([test_sample['wavelength']], dtype=torch.float32)
    true_val = test_sample['absorbance_target']
    
    # Predictions
    pred_untrained = model_untrained(c_s, c_t, wl)['absorbance'].item()
    pred_trained = model(c_s, c_t, wl)['absorbance'].item()
    
    print(f"   True value: {true_val:.4f}")
    print(f"   Untrained prediction: {pred_untrained:.4f} (error: {abs(pred_untrained - true_val):.4f})")
    print(f"   Trained prediction: {pred_trained:.4f} (error: {abs(pred_trained - true_val):.4f})")

print("\n" + "="*60)
print("DEMO COMPLETE!")
print(f"Total execution time: {time.time() - start_time:.1f} seconds")
print("="*60)

# Time estimates for full training
print("\n📊 TIME ESTIMATES FOR FULL TRAINING:")
print("-"*40)

time_per_sample = training_time / n_samples
total_samples = len(dataset)
samples_per_epoch_10pct = int(total_samples * 0.1)
samples_per_epoch_full = total_samples

print(f"Based on {time_per_sample:.3f} seconds per sample:")
print(f"\n10% sampling (recommended):")
print(f"  - 1 epoch: {samples_per_epoch_10pct * time_per_sample / 60:.1f} minutes")
print(f"  - 10 epochs: {samples_per_epoch_10pct * time_per_sample * 10 / 60:.1f} minutes")
print(f"  - 50 epochs: {samples_per_epoch_10pct * time_per_sample * 50 / 60:.1f} minutes")

print(f"\nFull data:")
print(f"  - 1 epoch: {samples_per_epoch_full * time_per_sample / 60:.1f} minutes")
print(f"  - 10 epochs: {samples_per_epoch_full * time_per_sample * 10 / 3600:.1f} hours")
print(f"  - 100 epochs: {samples_per_epoch_full * time_per_sample * 100 / 3600:.1f} hours")

print("\n💡 RECOMMENDATION:")
print("Use 10% sampling with 50-100 epochs for practical training")
print("This gives good results in 1-2 hours instead of days")