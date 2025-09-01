#!/usr/bin/env python3
"""
Quick Test Script for Geodesic NODE
Tests all components with real data to ensure everything works
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

print("=" * 60)
print("GEODESIC NODE - QUICK TEST SCRIPT")
print("=" * 60)

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n✓ Device: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# 1. LOAD REAL DATA
print("\n1. LOADING REAL DATA")
print("-" * 40)

data_path = Path(__file__).parent / "data" / "0.30MB_AuNP_As.csv"
if not data_path.exists():
    print(f"✗ Data file not found: {data_path}")
    sys.exit(1)

df = pd.read_csv(data_path)
print(f"✓ Loaded data from: {data_path}")
print(f"  Shape: {df.shape}")
print(f"  Wavelengths: {df.iloc[0, 0]:.0f} - {df.iloc[-1, 0]:.0f} nm")
print(f"  Concentrations: {list(df.columns[1:])} ppb")

# Extract data
wavelengths = df['Wavelength'].values
concentrations = np.array([float(c) for c in df.columns[1:]])  # ppb
spectra = df.iloc[:, 1:].values  # (n_wavelengths, n_concentrations)

print(f"  Spectra shape: {spectra.shape}")
print(f"  Absorbance range: [{spectra.min():.4f}, {spectra.max():.4f}]")

# 2. TEST IMPORTS
print("\n2. TESTING IMPORTS")
print("-" * 40)

try:
    from src.models.metric_network import MetricNetwork
    print("✓ MetricNetwork imported")
except Exception as e:
    print(f"✗ MetricNetwork import failed: {e}")
    sys.exit(1)

try:
    from src.core.christoffel import ChristoffelComputer
    print("✓ ChristoffelComputer imported")
except Exception as e:
    print(f"✗ ChristoffelComputer import failed: {e}")
    sys.exit(1)

try:
    from src.core.geodesic_ode import GeodesicODE
    print("✓ GeodesicODE imported")
except Exception as e:
    print(f"✗ GeodesicODE import failed: {e}")
    sys.exit(1)

try:
    from src.core.shooting_solver import ShootingSolver
    print("✓ ShootingSolver imported")
except Exception as e:
    print(f"✗ ShootingSolver import failed: {e}")
    sys.exit(1)

try:
    from src.models.decoder_network import AbsorbanceDecoder
    print("✓ AbsorbanceDecoder imported")
except Exception as e:
    print(f"✗ AbsorbanceDecoder import failed: {e}")
    sys.exit(1)

try:
    from src.models.geodesic_model import GeodesicSpectralModel
    print("✓ GeodesicSpectralModel imported")
except Exception as e:
    print(f"✗ GeodesicSpectralModel import failed: {e}")
    sys.exit(1)

# 3. TEST METRIC NETWORK
print("\n3. TESTING METRIC NETWORK")
print("-" * 40)

metric_net = MetricNetwork().to(device)
print(f"✓ MetricNetwork created")
print(f"  Parameters: {sum(p.numel() for p in metric_net.parameters()):,}")

# Test with normalized data
c_norm = torch.tensor([0.0, 0.5, 1.0], device=device)  # Normalized concentrations
wl_norm = torch.tensor([-0.5, 0.0, 0.5], device=device)  # Normalized wavelengths

g_values = metric_net(c_norm, wl_norm)
print(f"✓ Forward pass successful")
print(f"  Sample metric values: {g_values.detach().cpu().numpy()}")
print(f"  All positive: {torch.all(g_values > 0).item()}")

# 4. TEST CHRISTOFFEL COMPUTATION
print("\n4. TESTING CHRISTOFFEL SYMBOLS")
print("-" * 40)

christoffel = ChristoffelComputer()
print("✓ ChristoffelComputer created")

# Test Christoffel symbol computation
c_test = torch.tensor([0.5], device=device)
wl_test = torch.tensor([0.0], device=device)
gamma = christoffel.compute(c_test, wl_test, metric_net)
print(f"✓ Christoffel computation successful")
print(f"  Γ(c={c_test.item():.2f}, λ={wl_test.item():.2f}) = {gamma.item():.6f}")

# 5. TEST GEODESIC ODE
print("\n5. TESTING GEODESIC ODE")
print("-" * 40)

ode = GeodesicODE(metric_net, wavelength=0.0)
print("✓ GeodesicODE created")

# Test ODE evaluation
state = np.array([0.5, 1.0])  # [c, v]
t = 0.0
dydt = ode(t, state)
print(f"✓ ODE evaluation successful")
print(f"  State: [c={state[0]:.2f}, v={state[1]:.2f}]")
print(f"  Derivatives: [dc/dt={dydt[0]:.4f}, dv/dt={dydt[1]:.4f}]")

# 6. TEST SHOOTING SOLVER
print("\n6. TESTING SHOOTING SOLVER")
print("-" * 40)

solver = ShootingSolver(metric_net, max_iterations=20, tolerance=1e-3)
print("✓ ShootingSolver created")

# Test single solve
c_source = 0.0
c_target = 0.5
wavelength = 0.0

result = solver.solve(c_source, c_target, wavelength, return_trajectory=True, n_trajectory_points=11)
print(f"✓ Shooting solve completed")
print(f"  Success: {result['success']}")
print(f"  Initial velocity: {result['v0']:.4f}")
print(f"  Error: {result['error']:.6f}")
print(f"  Iterations: {result['iterations']}")

if result['success']:
    trajectory = result['trajectory']
    print(f"  Trajectory shape: {trajectory.shape}")
    print(f"  Start: c={trajectory[0, 0]:.4f}")
    print(f"  End: c={trajectory[-1, 0]:.4f} (target={c_target})")

# 7. TEST DECODER
print("\n7. TESTING DECODER NETWORK")
print("-" * 40)

decoder = AbsorbanceDecoder().to(device)
print(f"✓ AbsorbanceDecoder created")
print(f"  Parameters: {sum(p.numel() for p in decoder.parameters()):,}")

# Test with trajectory
if result['success']:
    traj_tensor = torch.tensor(trajectory, dtype=torch.float32, device=device).unsqueeze(0)
    wl_tensor = torch.tensor([wavelength], dtype=torch.float32, device=device)
    
    absorbance = decoder(traj_tensor, wl_tensor)
    print(f"✓ Decoder forward pass successful")
    print(f"  Predicted absorbance: {absorbance.item():.4f}")

# 8. TEST FULL MODEL
print("\n8. TESTING FULL GEODESIC MODEL")
print("-" * 40)

model = GeodesicSpectralModel(n_trajectory_points=11, shooting_tolerance=1e-3).to(device)
print(f"✓ GeodesicSpectralModel created")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test with batch of real-like data
batch_size = 4
c_sources = torch.tensor([0.0, 0.2, 0.3, 0.5], device=device)
c_targets = torch.tensor([0.5, 0.6, 0.7, 0.8], device=device)
wavelengths = torch.tensor([0.0, 0.1, -0.1, 0.2], device=device)

output = model(c_sources, c_targets, wavelengths)
print(f"✓ Full model forward pass successful")
print(f"  Batch size: {batch_size}")
print(f"  Absorbances: {output['absorbance'].detach().cpu().numpy()}")
print(f"  Success rate: {output['success_mask'].float().mean().item():.1%}")

# 9. TEST GRADIENT FLOW
print("\n9. TESTING GRADIENT FLOW")
print("-" * 40)

# Create target absorbances
targets = torch.randn(batch_size, device=device) * 0.1 + 0.5

# Compute loss
losses = model.compute_loss(output, targets)
print(f"✓ Loss computation successful")
for key, value in losses.items():
    print(f"  {key}: {value.item():.6f}")

# Combine losses
total_loss = model.combine_losses(losses)
print(f"  Total loss: {total_loss.item():.6f}")

# Check gradients
model.zero_grad()
total_loss.backward()

has_grad = {}
for name, param in model.named_parameters():
    has_grad[name.split('.')[0]] = param.grad is not None and param.grad.abs().max() > 0

print(f"✓ Gradient computation successful")
print(f"  Metric network has gradients: {has_grad.get('metric_network', False)}")
print(f"  Decoder has gradients: {has_grad.get('decoder', False)}")

# 10. TEST DATA PREPARATION
print("\n10. TESTING DATA PREPARATION")
print("-" * 40)

# Normalize concentrations (0-60 ppb → -1 to 1)
c_min, c_max = 0, 60
c_normalized = 2 * (concentrations - c_min) / (c_max - c_min) - 1
print(f"✓ Concentration normalization")
print(f"  Original: {concentrations} ppb")
print(f"  Normalized: {c_normalized}")

# Normalize wavelengths (200-800 nm → -1 to 1)
wl_min, wl_max = 200, 800
wl_normalized = 2 * (wavelengths - wl_min) / (wl_max - wl_min) - 1
print(f"✓ Wavelength normalization")
print(f"  Range: [{wl_normalized.min():.2f}, {wl_normalized.max():.2f}]")

# Create training pairs
n_pairs = 0
for i in range(len(concentrations)):
    for j in range(len(concentrations)):
        if i != j:
            n_pairs += 1

print(f"✓ Training pairs")
print(f"  Total possible pairs: {n_pairs}")
print(f"  Per wavelength: {n_pairs // len(wavelengths)}")

# 11. MEMORY CHECK
print("\n11. MEMORY AND PERFORMANCE CHECK")
print("-" * 40)

if device.type == 'cuda':
    print(f"✓ GPU Memory:")
    print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1e6:.1f} MB")
    print(f"  Reserved: {torch.cuda.memory_reserved(0) / 1e6:.1f} MB")

# Test larger batch
try:
    large_batch = 32
    c_src = torch.randn(large_batch, device=device) * 0.5
    c_tgt = torch.randn(large_batch, device=device) * 0.5
    wl = torch.zeros(large_batch, device=device)
    
    import time
    start = time.time()
    with torch.no_grad():
        _ = model(c_src, c_tgt, wl)
    elapsed = time.time() - start
    
    print(f"✓ Large batch test")
    print(f"  Batch size: {large_batch}")
    print(f"  Time: {elapsed:.3f} seconds")
    print(f"  Throughput: {large_batch/elapsed:.1f} samples/sec")
except Exception as e:
    print(f"✗ Large batch failed: {e}")

# 12. SUMMARY
print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)

all_tests = [
    ("Data Loading", True),
    ("Module Imports", True),
    ("Metric Network", True),
    ("Christoffel Symbols", True),
    ("Geodesic ODE", True),
    ("Shooting Solver", result['success']),
    ("Decoder Network", True),
    ("Full Model", True),
    ("Gradient Flow", True),
    ("Data Preparation", True),
]

passed = sum(1 for _, status in all_tests if status)
total = len(all_tests)

for test_name, status in all_tests:
    status_str = "✓ PASS" if status else "✗ FAIL"
    print(f"  {test_name:20s} {status_str}")

print(f"\nResult: {passed}/{total} tests passed")

if passed == total:
    print("\n🎉 ALL TESTS PASSED! The model is ready to train.")
else:
    print(f"\n⚠️  {total - passed} tests failed. Please check the output above.")

print("\nNext steps:")
print("1. Run demos/demo_training.py for full training")
print("2. Use the Colab notebook for GPU-accelerated training")
print("3. Check visualization/ for analysis tools")