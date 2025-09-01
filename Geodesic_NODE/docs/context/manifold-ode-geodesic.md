# TRUE 1D Geodesic Manifold-NODE Implementation Instructions

## Critical Context

**IMPORTANT**: This document describes how to implement a neural network that **actually solves geodesic equations**, not just pretends to. Previous attempts incorrectly claimed to use geodesics while only evolving latent states. This implementation truly solves the differential equation:

```
d²c/dt² = -Γ(c,λ)(dc/dt)²
```

Where:
- c is concentration (the variable we interpolate over)
- λ is wavelength (acts as a conditioning parameter)
- Γ is the Christoffel symbol computed from the learned metric
- t is the integration parameter ∈ [0,1]

## Problem Statement

### Given Data
- **Wavelengths**: 601 discrete measurements from 200-800 nm
- **Concentrations**: 6 measured values [0, 10, 20, 30, 40, 60] ppb
- **Absorbance Matrix**: 601 × 6 measurements
- **Total Training Points**: 3,606 (all wavelength-concentration pairs)

### Challenge
- Linear interpolation fails catastrophically at edge cases (R² = -34.13 at 60 ppb)
- Non-monotonic spectral response (gray → blue → gray) creates ambiguity
- Only 6 concentration points per wavelength (severe data sparsity)

### Solution
Learn a Riemannian metric g(c,λ) where geodesics in 1D concentration space provide optimal interpolation paths. Each wavelength defines its own geometry.

---

## Mathematical Foundation

### 1D Riemannian Geometry

For each wavelength λ, we have a 1D Riemannian manifold over concentration space c ∈ [0, 60] ppb.

**Metric Tensor**: g(c,λ) is a scalar function (since we're in 1D)
- Interpretation: g measures the "cost" of moving through concentration c at wavelength λ
- Large g = difficult to traverse (rapid spectral changes)
- Small g = easy to traverse (smooth spectral changes)

**Christoffel Symbol** (1D case):
```
Γ(c,λ) = ½ g⁻¹(c,λ) ∂g(c,λ)/∂c
```

**Geodesic Equation** (2nd order ODE):
```
d²c/dt² + Γ(c,λ)(dc/dt)² = 0
```

### Converting to First-Order System

To solve numerically, convert to first-order:
```
State vector: s = [c, v] where v = dc/dt

System of ODEs:
dc/dt = v
dv/dt = -Γ(c,λ)v²
```

### Boundary Value Problem

We need geodesics connecting two concentrations:
- **Initial condition**: c(0) = c_source
- **Final condition**: c(1) = c_target
- **Unknown**: Initial velocity v(0) that achieves c(1) = c_target

This requires a **shooting method** to find the correct initial velocity.

---

## Architecture Components

### Component 1: Metric Network

**Purpose**: Learn the Riemannian metric g(c,λ) that defines the geometry

**Input**: [c, λ] where both are normalized to [-1, 1]

**Architecture**:
```
Linear(2 → 64) + Tanh activation
Linear(64 → 128) + Tanh activation  
Linear(128 → 1) + no activation
Output: raw_metric
Final: g(c,λ) = softplus(raw_metric) + 0.1
```

**Constraints**:
- Must output g > 0 (enforced by softplus + 0.1)
- Should be smooth in c (will add regularization)
- Different behavior for different wavelengths

### Component 2: Christoffel Symbol Computer

**Purpose**: Compute Γ(c,λ) = ½ g⁻¹(c,λ) ∂g(c,λ)/∂c

**Implementation Options**:

**Option A - Finite Differences** (Recommended for stability):
```
ε = 1e-4
g_plus = metric_network(c + ε, λ)
g_minus = metric_network(c - ε, λ)
g_center = metric_network(c, λ)
dg_dc = (g_plus - g_minus) / (2ε)
Γ = 0.5 * dg_dc / g_center
```

**Option B - Automatic Differentiation**:
```
Enable gradients for c
g = metric_network(c, λ)
dg_dc = autograd.grad(g, c)
Γ = 0.5 * dg_dc / g
```

**Important**: Cache computations when possible to avoid redundant forward passes.

### Component 3: Geodesic ODE Function

**Purpose**: Define the ODE system for numerical integration

**State Vector**: [c, v] where v = dc/dt

**ODE System**:
```
Input: t (time), state=[c,v], λ (wavelength)
Output: d(state)/dt = [dc/dt, dv/dt]

Steps:
1. Extract c and v from state
2. Compute Γ(c,λ) using Component 2
3. Return [v, -Γv²]
```

**Critical**: This function will be called many times by the ODE solver, so efficiency matters.

### Component 4: Shooting Method Solver

**Purpose**: Solve the boundary value problem to find geodesics

**Problem**: Given c_source and c_target, find initial velocity v₀ such that integrating from [c_source, v₀] at t=0 reaches c_target at t=1.

**Algorithm**:
```
1. Define objective function:
   f(v₀) = (c_final(v₀) - c_target)²
   where c_final is obtained by integrating ODE with initial [c_source, v₀]

2. Minimize f(v₀) using:
   - Option A: Gradient-based optimization (requires differentiable ODE solver)
   - Option B: Gradient-free optimization (Powell, Nelder-Mead)
   - Option C: Simple bisection if monotonic

3. Return optimal v₀ and full trajectory
```

**Implementation Notes**:
- Start with reasonable initial guess: v₀ = (c_target - c_source)
- Add bounds to prevent extreme velocities
- Handle failure cases (no solution exists)

### Component 5: Absorbance Decoder

**Purpose**: Map geodesic path to absorbance prediction

**Input Options**:

**Option A - Endpoint Only**:
```
Input: [c_final, λ]
Architecture: 2 → 64 → 128 → 1
Output: absorbance
```

**Option B - Path Statistics**:
```
Input: [c_final, c_mean, path_length, max_velocity, λ]
Architecture: 5 → 64 → 128 → 1
Output: absorbance
```

**Option C - Full Path Encoding**:
```
Input: geodesic trajectory (11 points × 2 dims) + λ
Process: 1D CNN or RNN over trajectory
Output: absorbance
```

**Recommendation**: Start with Option A (simplest), upgrade if needed.

---

## Training Pipeline

### Data Preparation

1. **Load CSV**: Extract 601 wavelengths × 6 concentrations matrix
2. **Normalize**:
   - λ_norm = (λ - 500) / 300
   - c_norm = (c - 30) / 30  
   - A_norm = (A - mean(A)) / std(A)
3. **Create Training Tuples**: 
   - For concentration pairs: (c_source, c_target, λ, A_target)
   - Total: 6×5×601 = 18,030 pairs per epoch

### Training Loop

```
For each epoch:
    For each batch of (c_source, c_target, λ, A_target):
        
        1. Solve Geodesic BVP:
           - Use shooting method to find v₀
           - Integrate geodesic from [c_source, v₀] to get trajectory
           - Verify c(1) ≈ c_target (if not, optimization failed)
        
        2. Decode to Absorbance:
           - Extract features from geodesic path
           - Pass through decoder network
           - Get predicted absorbance A_pred
        
        3. Compute Loss:
           - Reconstruction: MSE(A_pred, A_target)
           - Metric smoothness: (∂²g/∂c²)²
           - Metric bounds: ReLU(-g + 0.01) + ReLU(g - 100)
           - Path efficiency: Minimize path length
           - Total: L_recon + λ₁L_smooth + λ₂L_bounds + λ₃L_path
        
        4. Backpropagation:
           - Backprop through decoder
           - Backprop through ODE solver (using adjoint method)
           - Backprop through metric network
           - Update all parameters
```

### Optimizer Configuration

- **Algorithm**: Adam
- **Learning Rates**:
  - Metric Network: 5e-4 (careful updates)
  - Decoder Network: 1e-3 (faster learning)
- **Gradient Clipping**: max_norm = 1.0
- **Scheduler**: Cosine annealing over 500 epochs

### Loss Weights

Start with:
- λ₁ (smoothness) = 0.01
- λ₂ (bounds) = 0.001  
- λ₃ (path) = 0.001

Adjust based on relative magnitudes during training.

---

## Implementation Checklist

### Phase 1: Core Components
- [ ] Implement Metric Network with proper initialization
- [ ] Implement Christoffel Symbol computation (finite differences)
- [ ] Implement Geodesic ODE function
- [ ] Test ODE integration with fixed metric (sanity check)

### Phase 2: Boundary Value Problem
- [ ] Implement shooting method objective function
- [ ] Add optimization routine (start with scipy.optimize)
- [ ] Test BVP solver on known solutions
- [ ] Handle edge cases (no solution, multiple solutions)

### Phase 3: Full Model
- [ ] Implement Absorbance Decoder (start simple)
- [ ] Connect all components
- [ ] Verify forward pass works
- [ ] Check gradient flow

### Phase 4: Training
- [ ] Set up data loading and normalization
- [ ] Implement training loop with BVP solving
- [ ] Add loss functions and regularization
- [ ] Implement checkpointing

### Phase 5: Validation
- [ ] Leave-one-out cross-validation setup
- [ ] Compute 23 metrics (focus on 6 key ones)
- [ ] Test interpolation at intermediate concentrations
- [ ] Compare against linear baseline

---

## Critical Implementation Notes

### What Makes This Different

1. **Actually Solves Geodesics**: The ODE explicitly computes d²c/dt² = -Γ(c,λ)(dc/dt)², not some latent approximation.

2. **Proper Boundary Conditions**: Uses shooting method to ensure geodesics connect source to target concentrations.

3. **Physical Interpretation**: The metric g(c,λ) has clear meaning - it measures how "difficult" it is to move through concentration space.

4. **No Overfitting**: Single shared network with 3,606 training points, not 601 separate models.

### Common Pitfalls to Avoid

1. **Don't Skip the Shooting Method**: You can't just integrate from c_source with arbitrary velocity and hope to reach c_target.

2. **Don't Ignore Numerical Stability**: Geodesic equations can be stiff. Start with small integration steps.

3. **Don't Forget Normalization**: All inputs should be in [-1, 1] for stable training.

4. **Don't Use Wrong State Vector**: State is [c, v], not [z, c, λ] or other variations.

5. **Don't Pretend to Solve Geodesics**: If you're just evolving a latent state, that's not a geodesic.

### Debugging Strategy

1. **Start with Fixed Metric**: Use g(c,λ) = 1 (Euclidean) to verify ODE solver works.

2. **Test on Single Wavelength**: Fix λ and ensure 1D geodesics work.

3. **Visualize Geodesics**: Plot c(t) trajectories to ensure they're smooth.

4. **Monitor Christoffel Symbols**: They should be smooth and bounded.

5. **Check Conservation**: In regions where g is constant, geodesics should be straight lines.

---

## Expected Behavior

### During Training
- Loss should decrease steadily
- Geodesics should become smoother
- Metric should develop structure (large values where spectrum changes rapidly)
- Shooting method should converge faster as metric improves

### Validation Results
- R² Score: Should improve from -34.13 to > 0.7
- Peak λ Error: Should reduce from 459 nm to < 20 nm  
- MAPE: Should drop from 100.7% to < 20%
- Geodesics at 40-60 ppb: Should show interesting curvature

### Physical Interpretation
- High g(c,λ): Rapid spectral changes (like near 50-60 ppb transition)
- Low g(c,λ): Smooth spectral behavior
- Curved geodesics: Optimal paths that "avoid" difficult regions
- Straight geodesics: Direct paths through easy regions

---

## Final Notes

This implementation represents a mathematically rigorous approach to spectral interpolation using Riemannian geometry. The key innovation is that we're actually solving the geodesic equation, not approximating it with neural networks.

The geodesics provide a principled way to interpolate through non-monotonic responses by learning the underlying geometry of the spectral manifold. Each wavelength gets its own 1D geometry, but all wavelengths share the same neural network parameters, preventing overfitting while allowing wavelength-specific behavior.

Success depends on:
1. Correct implementation of the geodesic ODE
2. Stable shooting method for BVP solving  
3. Proper backpropagation through the ODE solver
4. Careful regularization of the metric function

When implemented correctly, this approach should dramatically outperform linear interpolation, especially at the problematic edge concentrations where the spectral response is most non-linear.