# Riemannian Manifold-NODE Implementation Instructions

## Overview

This document provides comprehensive instructions for implementing a Riemannian Manifold Neural Ordinary Differential Equation (Manifold-NODE) model for spectral interpolation. The model learns to interpolate UV-Vis absorbance spectra across different arsenic concentrations by learning the underlying manifold geometry.

## Table of Contents
1. [Mathematical Foundation](#mathematical-foundation)
2. [Data Preprocessing](#data-preprocessing)
3. [Model Architecture](#model-architecture)
4. [Loss Functions](#loss-functions)
5. [Training Strategy](#training-strategy)
6. [Validation Protocol](#validation-protocol)
7. [Stability Measures](#stability-measures)
8. [Implementation Checklist](#implementation-checklist)

---

## 1. Mathematical Foundation

### 1.1 Problem Setup

Given:
- Wavelengths: λ ∈ [200, 800] nm (601 discrete points)
- Concentrations: c ∈ {0, 10, 20, 30, 40, 60} ppb (6 values)
- Absorbance: A(λ, c) ∈ ℝ⁺

Goal: Learn a continuous function f: (λ, c) → A that respects the manifold geometry of the spectral response.

### 1.2 Riemannian Manifold

The spectral data lives on a 2D Riemannian manifold ℳ embedded in 3D space:

```
ℳ = {(λ, c, A(λ, c)) : λ ∈ [200, 800], c ∈ [0, 60]}
```

### 1.3 Metric Tensor

At each point (λ, c), the metric tensor g is a 2×2 positive definite matrix:

```
g = [g₁₁  g₁₂]
    [g₁₂  g₂₂]
```

The infinitesimal distance on the manifold:
```
ds² = g₁₁dλ² + 2g₁₂dλdc + g₂₂dc²
```

### 1.4 Christoffel Symbols

The Christoffel symbols Γᵏᵢⱼ encode the connection on the manifold:

```
Γᵏᵢⱼ = ½ gᵏˡ(∂gᵢₗ/∂xʲ + ∂gⱼₗ/∂xⁱ - ∂gᵢⱼ/∂xˡ)
```

### 1.5 Geodesic Equation

A geodesic γ(t) = (λ(t), c(t)) satisfies:

```
d²xᵏ/dt² + Γᵏᵢⱼ(dxⁱ/dt)(dxʲ/dt) = 0
```

---

## 2. Data Preprocessing

### 2.1 Data Loading

1. Load the CSV file containing spectral data
2. Extract:
   - Wavelength column (601 values)
   - Concentration columns (6 columns)
   - Create absorbance matrix (601 × 6)

### 2.2 Normalization

Normalize all inputs and outputs for numerical stability:

#### Input Normalization:
```
λ_normalized = (λ - 500) / 300  # Maps to approximately [-1, 1]
c_normalized = (c - 30) / 30    # Maps to [-1, 1]
```

#### Output Normalization:
```
A_mean = mean(absorbance_matrix)
A_std = std(absorbance_matrix)
A_normalized = (A - A_mean) / A_std
```

Store normalization parameters for later denormalization.

### 2.3 Dataset Creation

Create tuples of (λ, c, A) for all combinations:
- Total data points: 601 × 6 = 3,606
- Each data point: {λ_norm, c_norm, A_norm, λ_raw, c_raw}

### 2.4 DataLoader Configuration

- Use full batch training (batch_size = 3606) due to small dataset
- Shuffle data each epoch
- No data augmentation needed

---

## 3. Model Architecture

### 3.1 Overall Structure

The model consists of four main components:

1. **Encoder**: Maps (λ, c) to initial latent state z₀
2. **Metric Network**: Computes the metric tensor g(λ, c)
3. **Geodesic ODE**: Evolves latent state following geodesics
4. **Decoder**: Maps final latent state z₁ to absorbance A

### 3.2 Encoder Network

Architecture:
```
Input: [λ, c] (2D)
Layer 1: Linear(2 → 64) + ReLU
Layer 2: Linear(64 → 128) + ReLU
Layer 3: Linear(128 → 32)
Output: z₀ (32D latent state)
```

### 3.3 Metric Network

Architecture:
```
Input: [λ, c] (2D)
Layer 1: Linear(2 → 64) + Tanh
Layer 2: Linear(64 → 128) + Tanh
Layer 3: Linear(128 → 4)
Output: [L₁₁, L₂₁, L₂₂, dummy] for Cholesky parameterization
```

#### Metric Tensor Construction:
```
L₁₁ = softplus(output[0]) + 0.1  # Ensure > 0
L₂₁ = output[1]
L₂₂ = softplus(output[2]) + 0.1  # Ensure > 0

g₁₁ = L₁₁²
g₁₂ = L₁₁ × L₂₁
g₂₂ = L₂₁² + L₂₂²
```

This ensures positive definiteness of the metric tensor.

### 3.4 Geodesic ODE Function

The ODE function evolves the state [z, λ, c] over time t ∈ [0, 1].

Architecture for latent dynamics:
```
Input: [z, λ, c] (34D)
Layer 1: Linear(34 → 64) + Tanh
Layer 2: Linear(64 → 32)
Output: dz/dt (32D)
```

#### ODE Evolution:
```
State = [z (32D), λ (1D), c (1D)]
dState/dt = [dz/dt, 0, 0]  # λ and c remain fixed
```

### 3.5 Decoder Network

Architecture:
```
Input: z₁ (32D)
Layer 1: Linear(32 → 64) + ReLU
Layer 2: Linear(64 → 128) + ReLU
Layer 3: Linear(128 → 1)
Output: A_raw
Final: A = softplus(A_raw)  # Ensure non-negative
```

### 3.6 Weight Initialization

- Use Xavier normal initialization with gain=0.5 for all linear layers
- Initialize biases to zero
- This small initialization helps prevent exploding gradients

---

## 4. Loss Functions

### 4.1 Total Loss Function

```
L_total = L_reconstruction + λ₁L_smoothness + λ₂L_path + λ₃L_metric
```

Recommended weights: λ₁=0.01, λ₂=0.001, λ₃=0.001

### 4.2 Reconstruction Loss

Mean squared error between predicted and true absorbance:
```
L_reconstruction = (1/N) Σᵢ(A_pred[i] - A_true[i])²
```

### 4.3 Smoothness Loss

Penalize second derivatives to ensure smooth output:
```
L_smoothness = Σᵢ(A[i+1] - 2A[i] + A[i-1])²
```

Compute separately for wavelength and concentration dimensions.

### 4.4 Path Length Loss

Regularize the length of paths in latent space:
```
L_path = (1/(T-1)) Σₜ ||z[t+1] - z[t]||₂
```

Where T is the number of integration steps.

### 4.5 Metric Regularization Loss

Ensure the metric tensor remains well-behaved:
```
det(g) = g₁₁g₂₂ - g₁₂²

L_metric = ReLU(-det(g) + 0.01) + ReLU(det(g) - 100) + 0.1×ReLU(trace(g) - 10)
```

This prevents metric collapse or explosion.

---

## 5. Training Strategy

### 5.1 Optimizer Configuration

Use Adam optimizer with different learning rates:
- Encoder parameters: lr = 1e-3
- Decoder parameters: lr = 1e-3
- Metric network parameters: lr = 5e-4
- ODE function parameters: lr = 5e-4

### 5.2 Learning Rate Schedule

Use cosine annealing schedule:
```
lr(t) = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × t / T))
```

Where T is total epochs (500).

### 5.3 ODE Solver Configuration

For training:
- Method: 'rk4' (4th order Runge-Kutta)
- Fixed step size: 0.1
- Time points: [0, 0.1, 0.2, ..., 1.0]

For inference:
- Method: 'dopri5' (adaptive Dormand-Prince)
- Tolerances: rtol=1e-3, atol=1e-4

### 5.4 Gradient Clipping

Apply gradient clipping with max_norm = 1.0 to prevent instability:
```
torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
```

### 5.5 Training Loop

```
For epoch in range(500):
    1. Forward pass through entire dataset
    2. Compute all loss components
    3. Backward pass
    4. Check for NaN gradients (zero them if found)
    5. Clip gradients
    6. Optimizer step
    7. Learning rate scheduler step
    8. Log metrics every 10 epochs
    9. Save checkpoint every 50 epochs
```

### 5.6 Early Stopping

Monitor validation loss on middle concentrations (20, 30 ppb):
- If no improvement for 50 epochs, stop training
- Keep best model based on validation loss

---

## 6. Validation Protocol

### 6.1 Leave-One-Out Cross-Validation

For each concentration c_holdout ∈ {0, 10, 20, 30, 40, 60}:
1. Train on remaining 5 concentrations
2. Test on held-out concentration
3. Calculate all 23 metrics
4. Focus on 6 key metrics:
   - R² Score
   - Peak λ Error
   - MAPE
   - SAM (Spectral Angle Mapper)
   - Wasserstein Distance
   - Power Ratio

### 6.2 Interpolation Testing

Test at intermediate concentrations not in training set:
- Test concentrations: {5, 15, 25, 35, 45, 55} ppb
- Generate full spectra for each
- Verify smoothness and physical validity

### 6.3 Metric Calculations

For each metric, compare against baseline linear interpolation:
```
Improvement_ratio = |Metric_baseline| / |Metric_model|
```

Target improvements:
- R² Score: From -34 to > 0.7
- Peak λ Error: From 459nm to < 20nm
- MAPE: From 100% to < 20%

---

## 7. Stability Measures

### 7.1 Numerical Stability Checks

#### Metric Tensor Stability:
- Check determinant: 0.01 < det(g) < 100
- Check condition number: cond(g) < 1000
- If violated, reset metric network weights

#### Gradient Health:
- Check for NaN/Inf gradients each iteration
- Monitor gradient norms (should be < 10)
- If unstable, reduce learning rate

### 7.2 Christoffel Symbol Computation

Use finite differences instead of autograd for stability:
```
∂g/∂λ ≈ (g(λ+ε, c) - g(λ-ε, c)) / (2ε)
∂g/∂c ≈ (g(λ, c+ε) - g(λ, c-ε)) / (2ε)
```

Use ε = 1e-4.

### 7.3 ODE Integration Stability

- Start with fewer integration steps (5-10)
- Use fixed-step solver during training
- Monitor trajectory divergence
- If divergence detected, reduce step size

### 7.4 Loss Component Balancing

Monitor relative magnitudes of loss components:
- If one dominates, adjust weights
- Typical ratio: L_recon : L_smooth : L_path : L_metric = 1000 : 10 : 1 : 1

---

## 8. Implementation Checklist

### 8.1 Data Pipeline
- [ ] Load CSV data correctly
- [ ] Verify data shape (601 × 6)
- [ ] Apply normalization
- [ ] Store normalization parameters
- [ ] Create DataLoader

### 8.2 Model Components
- [ ] Implement Encoder (2 → 32)
- [ ] Implement Metric Network with Cholesky parameterization
- [ ] Implement Geodesic ODE function
- [ ] Implement Decoder (32 → 1)
- [ ] Apply proper weight initialization

### 8.3 Loss Functions
- [ ] Implement reconstruction loss
- [ ] Implement smoothness penalty
- [ ] Implement path length regularization
- [ ] Implement metric regularization
- [ ] Balance loss weights

### 8.4 Training
- [ ] Configure optimizer with different learning rates
- [ ] Implement learning rate scheduler
- [ ] Set up ODE solver
- [ ] Add gradient clipping
- [ ] Implement checkpointing
- [ ] Add early stopping

### 8.5 Validation
- [ ] Implement leave-one-out validation
- [ ] Add interpolation testing
- [ ] Calculate all metrics
- [ ] Create visualization functions
- [ ] Compare with baseline

### 8.6 Stability
- [ ] Add metric tensor checks
- [ ] Implement gradient monitoring
- [ ] Use finite differences for derivatives
- [ ] Add loss component monitoring
- [ ] Implement recovery mechanisms

---

## 9. Expected Challenges and Solutions

### Challenge 1: Limited Data (only 3,606 points)
**Solution**: Strong regularization, simple architecture, early stopping

### Challenge 2: Non-monotonic response at 50-60 ppb
**Solution**: Metric tensor will learn to "stretch" space in this region

### Challenge 3: Numerical instability in ODE
**Solution**: Fixed-step solver, gradient clipping, small learning rates

### Challenge 4: Overfitting
**Solution**: Smoothness penalties, dropout in encoder/decoder, ensemble models

### Challenge 5: Poor edge performance (0 and 60 ppb)
**Solution**: Metric learning will adapt to edge geometry

---

## 10. Final Notes

### Key Success Factors:
1. **Proper normalization** - Critical for numerical stability
2. **Balanced loss weights** - Prevent any component from dominating
3. **Gradient clipping** - Essential for stable ODE training
4. **Metric regularization** - Keeps geometry well-behaved
5. **Patience** - Model may need 300+ epochs to converge

### Expected Training Time:
- GPU: 1-2 hours for 500 epochs
- CPU: 4-6 hours for 500 epochs

### Memory Requirements:
- Model parameters: ~500K
- Batch data: ~100KB
- ODE solver overhead: ~1MB
- Total: < 10MB (easily fits in memory)

### Output Files to Generate:
1. `manifold_node_model.pth` - Trained model weights
2. `training_log.csv` - Loss components over epochs
3. `validation_metrics.csv` - All 23 metrics for each holdout
4. `interpolation_results.npy` - Predictions at intermediate concentrations
5. `manifold_visualization.html` - Interactive 3D plot of learned manifold

---

## Implementation Order

1. **Start with data pipeline** - Get data loading working correctly
2. **Build basic model** - Encoder → Simple ODE → Decoder (no geodesics yet)
3. **Add reconstruction loss** - Get basic training working
4. **Add metric network** - Introduce geometric structure
5. **Add geodesic components** - Full Riemannian geometry
6. **Add all loss terms** - Complete training objective
7. **Implement validation** - Leave-one-out testing
8. **Add stability measures** - Make training robust
9. **Fine-tune hyperparameters** - Optimize performance
10. **Generate final results** - Metrics, visualizations, analysis

This incremental approach ensures each component works before adding complexity.