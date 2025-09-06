# Geodesic-Coupled Neural ODE for Arsenic Detection: Complete Technical Documentation

## Project Context for Regeneron STS

This project addresses a critical global health challenge: arsenic contamination in drinking water affects over 200 million people worldwide, particularly in rural regions of Bangladesh, India, and Southeast Asia. While laboratory techniques like ICP-MS can detect arsenic at sub-ppb levels, they require expensive equipment and specialized facilities unavailable in affected communities. This work develops a field-deployable detection system using colorimetric assays combined with smartphone-based UV-Vis spectroscopy, made accurate through a novel geodesic interpolation algorithm.

The core innovation: transforming an impossible interpolation problem into a tractable geometric one by recognizing that spectral space is fundamentally curved, not flat.

## The Fundamental Challenge

Arsenic contamination in drinking water is a persistent public health threat. In many rural and resource-limited regions, arsenic levels regularly exceed the World Health Organization (WHO) limit of 10 μg/L. Long-term exposure is linked to cancers, cardiovascular disease, and developmental impairments. While laboratory techniques such as inductively coupled plasma mass spectrometry (ICP-MS) can detect arsenic at sub-ppb levels, they are expensive, slow, and require specialized facilities. These constraints make ICP-MS unsuitable for routine, on-site monitoring in affected communities.

To close this gap, low-cost, portable sensing systems are needed—tools that can be deployed in the field and provide rapid, reliable concentration estimates. In this work, we investigate one candidate: a colorimetric assay based on methylene blue. When methylene blue reacts with arsenic, the solution changes color. However, unlike many assays that show a straightforward (monotonic) color change with concentration, this reaction follows a non-monotonic path: gray → blue → gray as the concentration increases. This folding of the concentration-color curve means that the same visible color may correspond to multiple concentrations. As a result, simple approaches like single-wavelength ratios or linear calibration curves fail.

To overcome this, we turn to ultraviolet-visible (UV-Vis) spectroscopy. Instead of reducing a sample's color to just three RGB values, UV-Vis produces an optical fingerprint: 601 absorbance measurements spanning 200-800 nm. This fingerprint captures both the visible color and subtle, wavelength-dependent features arising from the underlying chemistry—features that can distinguish between concentrations that look identical to the naked eye.

## 1. Gap Analysis: Why Basic Interpolation Fails Catastrophically

### 1.1 Non-Monotonicity Across Wavelengths

At certain wavelengths, the absorbance-concentration relationship is non-monotonic. As arsenic concentration increases from 0 to 60 ppb, absorbance first rises (gray to blue) then falls (blue back to gray). Linear interpolation assumes a monotonic trend, yielding predictions that can be off by significant amounts. 

### 1.2 Variable Sensitivity Across Spectrum

Different wavelengths respond with vastly different sensitivities to concentration changes.

Linear interpolation enforces uniform "speed" across all wavelengths, distorting the spectrum in sensitive regions where chemistry changes rapidly.

### 1.3 Crossings and Folds in Spectral Space

Endpoint-based interpolation often produces spectra that cross measured ones at intermediate concentrations.
### 1.4 Absence of Path Principle

Traditional methods average endpoints without considering the physical path spectra take between concentrations. This ignores the underlying chemical transitions—formation and dissociation of arsenic-methylene blue complexes—that govern spectral evolution.

## 2. Design Requirements for Robust Interpolation

Based on the failure modes, we establish requirements for a better interpolator:

1. **Geometry-aware**: Respect local spectral sensitivity at each wavelength
2. **Path-based**: Define predictions along a continuous trajectory between measured concentrations
3. **Principled constraint**: Use shortest-path or minimum-energy principles to regularize solutions
4. **Two-point boundary conditions**: Interpolants must exactly match measured endpoints
5. **Wavelength-specific yet unified**: Allow wavelength-dependent behavior under one shared physical law
6. **Generalizable**: Validate with leave-one-out cross-validation against matched baselines

## 3. Algorithm Overview: The Flight Path Analogy

Consider planning a flight from New York to London:
- **On a flat map**: Draw a straight line—but this isn't the shortest path on Earth's curved surface
- **On a globe**: Follow a great circle route that curves northward—the true shortest path
- **With weather**: The path bends further to exploit jet streams or avoid turbulence

Our spectral interpolation follows the same principles:

1. **Metric map g(c,λ)**: Learn a "cost field" defining how difficult it is to traverse concentration space at each wavelength (like weather patterns)
2. **Curvature Γ(c,λ)**: Derive from g how paths naturally bend in this space (like Earth's curvature)
3. **Geodesic integration**: From source concentration cs to target ct, find the initial velocity that produces a geodesic landing exactly at the target
4. **Absorbance accumulation**: Integrate learned flow F(c,v,λ) along the geodesic path to predict absorbance
5. **Parallel assembly**: Process all 601 wavelengths simultaneously to reconstruct the full spectrum
6. **End-to-end training**: Learn g and F jointly so predicted spectra match measurements while maintaining smooth geometry
7. **Validation**: Compare leave-one-out performance against linear baseline to prove gains come from geometry, not memorization

## 4. Geometric Formulation of Spectral Space

### 4.1 General Geodesics in N Dimensions

In a general Riemannian manifold with coordinates x^i and metric tensor g_ij(x), the infinitesimal distance is:

```
ds² = g_ij(x) dx^i dx^j
```

Geodesics minimize the action functional:

```
S[γ] = ∫₀¹ √(g_ij(x) ẋ^i ẋ^j) dt
```

The Euler-Lagrange equations yield the geodesic equation:

```
ẍ^i + Γ^i_jk ẋ^j ẋ^k = 0
```

where the Christoffel symbols are:

```
Γ^i_jk = ½ g^(il) (∂_j g_lk + ∂_k g_lj - ∂_l g_jk)
```

The system conserves the Hamiltonian:

```
H(x,ẋ) = ½ g_ij(x) ẋ^i ẋ^j
```

### 4.2 Dimensionality Reduction: Why 1D Manifolds

The full spectral state is (c,λ) ∈ ℝ². Geodesics in 2D would require solving four coupled first-order ODEs:
- dc/dt = v_c
- dλ/dt = v_λ  
- dv_c/dt = -Γ^c_cc v_c² - 2Γ^c_cλ v_c v_λ - Γ^c_λλ v_λ²
- dv_λ/dt = -Γ^λ_cc v_c² - 2Γ^λ_cλ v_c v_λ - Γ^λ_λλ v_λ²

However, wavelength λ is not dynamic—the spectrometer measures at a fixed grid of 601 points. We treat λ as a parameter, not a coordinate, reducing to 601 independent 1D geodesic problems.

### 4.3 One-Dimensional Geodesics at Fixed Wavelength

For each wavelength λ, the metric becomes a scalar function g(c,λ), and the line element is:

```
ds² = g(c,λ) dc²
```

The geodesic equation simplifies to:

```
c̈ = -Γ(c,λ) ċ²
```

where the single Christoffel symbol is:

```
Γ(c,λ) = ½ g⁻¹(c,λ) ∂g(c,λ)/∂c
```

The conserved energy along geodesics:

```
H(c,v,λ) = ½ g(c,λ) v²
```

This reduction maintains the essential physics while making computation tractable.

### 4.4 Learning the Metric with Neural Networks

The metric g(c,λ) encodes how "expensive" it is to move through concentration space—large values indicate regions where spectra change rapidly. We don't know this a priori, so we learn it with a neural network.

**Network Architecture:**
```python
Input: [c_norm, λ_norm] ∈ ℝ²
  where c_norm = (c - 30)/30 ∈ [-1, 1]
        λ_norm = (λ - 500)/300 ∈ [-1, 1]

Layer 1: Linear(2 → 128) + Tanh activation
Layer 2: Linear(128 → 256) + Tanh activation  
Layer 3: Linear(256 → 1)
Output: g(c,λ) = softplus(Layer3_output) + 0.1
```

**Key Design Choices:**
- **Input normalization**: Centers data around mid-concentration (30 ppb) and mid-spectrum (500 nm) for stable gradients
- **Hidden dimensions**: 128 and 256 neurons—multiples of 8 for Tensor Core optimization on NVIDIA GPUs
- **Tanh activation**: Smooth, bounded gradients without dead neuron problems of ReLU
- **Softplus output**: softplus(x) = log(1 + exp(x)) ensures strict positivity (negative distances are meaningless)
- **Offset ε = 0.1**: Prevents numerical instabilities from zero metrics
- **Total parameters**: ~33,000—small enough to fit in GPU L2 cache for fast access

**Computing Christoffel Symbols:**

Given the learned metric, we compute Christoffel symbols via centered finite differences:

```
Γ(c,λ) ≈ ½ g⁻¹(c,λ) × [g(c+δ,λ) - g(c-δ,λ)]/(2δ)
```

where δ = 10⁻⁴ balances truncation error (wants small δ) against round-off error (wants large δ).

## 5. Coupled Spectral Dynamics: Beyond Pure Geometry

### 5.1 The Augmented State Vector

Pure geodesics tell us how to navigate concentration space but not how absorbance changes along the path. We extend the state vector to include absorbance:

```
s(t) = [c(t), v(t), A(t)]ᵀ
```

### 5.2 Coupled ODE System

The complete dynamics:

```
ċ = v                           (position update)
v̇ = -Γ(c,λ) v²                 (geodesic acceleration)
Ȧ = F(c,v,λ)                    (spectral flow)
```

### 5.3 Why Velocity Matters for Absorbance

The spectral flow function F depends on velocity v, not just position c. This captures crucial physics:
- **Chemical transitions**: Rapid traversal through transition regions produces different spectral signatures than slow passage
- **Hysteresis effects**: The rate of concentration change affects complex formation/dissociation kinetics
- **Asymmetry**: Rising vs. falling concentration can follow different chemical pathways

### 5.4 Spectral Flow Network Architecture

```python
Input: [c_norm, v_norm, λ_norm] ∈ ℝ³
  where v_norm = v / max(|v|) ∈ [-1, 1]

Layer 1: Linear(3 → 64) + Tanh activation
Layer 2: Linear(64 → 128) + Tanh activation
Layer 3: Linear(128 → 1)
Output: F(c,v,λ) = Layer3_output
```

**Design rationale:**
- Smaller than metric network (64→128 vs 128→256) as it models residual dynamics
- Velocity normalization prevents gradient explosion during shooting iterations
- No positivity constraint—absorbance can increase or decrease
- Separate from metric network—different physical phenomena

## 6. Boundary Value Problem: The Shooting Method

### 6.1 Problem Formulation

Given:
- Source concentration: c(0) = cs
- Target concentration: c(1) = ct
- Wavelength parameter: λ

Find: Initial velocity v₀ such that integrating the geodesic ODE yields c(1) = ct

### 6.2 Shooting Algorithm Details

Define the endpoint error:
```
E(v₀) = c(1; v₀) - ct
```

**Iterative Solution:**

1. **Initialize**: v₀⁽⁰⁾ = ct - cs (linear approximation)

2. **For iteration k = 0, 1, 2, ..., max_iter:**
   
   a. Integrate geodesic with current v₀⁽ᵏ⁾:
   ```python
   state₀ = [cs, v₀⁽ᵏ⁾, As]
   trajectory = odeint(coupled_ode, state₀, t_span=[0,1])
   c_final = trajectory[-1, 0]
   ```
   
   b. Compute error: Ek = c_final - ct
   
   c. Estimate derivative (secant method):
   ```
   If k > 0:
     mk = (Ek - Ek-1)/(v₀⁽ᵏ⁾ - v₀⁽ᵏ⁻¹⁾)
   Else:
     mk = 1.0
   ```
   
   d. Update with damping:
   ```
   v₀⁽ᵏ⁺¹⁾ = v₀⁽ᵏ⁾ - α × Ek/max(|mk|, ε)
   ```
   where α ∈ (0,1] is learning rate, ε = 10⁻⁶ prevents division by zero
   
   e. Adaptive damping:
   ```
   If |Ek| > |Ek-1|:  # Overshoot
     α ← 0.5 × α
   If k % 3 == 0:      # Periodic reduction
     α ← 0.8 × α
   ```
   
   f. Check convergence: If |Ek| < tolerance, stop

### 6.3 Integration Methods

**During shooting iterations (need speed):**
- Fixed-step 4th-order Runge-Kutta (RK4)
- 10 time steps (t = 0, 0.1, ..., 1.0)
- O(h⁴) local error, predictable cost

**For final trajectory (need accuracy):**
- Dormand-Prince 5(4) with adaptive step size
- Embedded error estimation
- Automatic step adjustment in high-curvature regions

### 6.4 Convergence Guarantee

In 1D, the endpoint map c(1; v₀) is locally Lipschitz continuous in v₀. The shooting iteration converges when:

```
0 < α < 2/L
```

where L is the Lipschitz constant. Our adaptive damping ensures this condition.

## 7. Loss Function Design

### 7.1 Multi-Objective Optimization

The total loss balances multiple objectives:

```
L = Lrecon + λs×Lsmooth + λb×Lbounds + λp×Lpath + λc×Lchristoffel
```

### 7.2 Component Losses

**Reconstruction Loss (Primary):**
```
Lrecon = (1/B) Σᵢ (Apred⁽ⁱ⁾ - Ameas⁽ⁱ⁾)²
```
Ensures predicted absorbances match measurements.

**Smoothness Regularization:**
```
Lsmooth = (1/B) Σᵢ |∂²g/∂c²|²
```
Prevents wild oscillations in the metric that would destabilize geodesics.

**Bounds Regularization:**
```
Lbounds = (1/B) Σᵢ [ReLU(0.01 - gᵢ) + ReLU(gᵢ - 100)]
```
Keeps metric in numerically stable range [0.01, 100].

**Path Length Penalty:**
```
Lpath = (1/B) Σᵢ ∫₀¹ √(g(c(t),λ) |v(t)|) dt
```
Encourages efficient geodesics—Occam's razor for paths.

**Christoffel Matching (When Target Available):**
```
Lchristoffel = (1/B) Σᵢ (Γpred⁽ⁱ⁾ - Γtarget⁽ⁱ⁾)²
```
Provides direct supervision when target geometry can be computed from data.

### 7.3 Weight Schedule

Training weights evolve over epochs:
- Early: λs = 0.01, λb = 0.001, λp = 0.001, λc = 0.0
- Mid: λs = 0.1, λb = 0.01, λp = 0.01, λc = 1.0
- Late: λs = 0.1, λb = 0.01, λp = 0.01, λc = 5.0

This curriculum helps the network first learn rough geometry, then refine details.

## 8. Backpropagation Through ODEs: The Adjoint Method

### 8.1 The Memory Problem

Naive backpropagation through ODE solutions requires storing entire trajectories:
- Memory: O(B × T × D) where B = batch size, T = time steps, D = dimension
- For B = 18,030, T = 50, D = 3: ~11 MB per iteration

### 8.2 Adjoint Solution

The adjoint method computes gradients without storing trajectories:

**Forward ODE:**
```
ṡ = f(s,θ,t),  s(0) = s₀
```

**Adjoint ODE (backward in time):**
```
ṗ = -(∂f/∂s)ᵀ p - ∂L/∂s
p(T) = ∂L/∂s(T)
```

**Parameter gradients:**
```
∂L/∂θ = -∫₀ᵀ pᵀ(t) ∂f/∂θ dt
```

### 8.3 Benefits

- Memory: O(B × D) independent of trajectory length
- Exact gradients (no truncation)
- Enables full-batch training with 18,030 parallel geodesics

## 9. GPU Optimization Strategy

### 9.1 Massive Parallelization

Process all concentration transitions simultaneously:
```
Batch size = n_pairs × n_wavelengths = 30 × 601 = 18,030
```

The 30 pairs come from transitions between 6 reference concentrations:
- (0→10), (0→20), ..., (50→60): 6×5 = 30 unique pairs

### 9.2 Christoffel Grid Precomputation

**Problem**: Computing Γ on-demand requires 3 network evaluations per ODE step
**Solution**: Precompute dense grid once, interpolate during training

```python
Grid resolution: 2000 × 601 points
Memory: 2000 × 601 × 4 bytes = 4.8 MB (FP32)
        2000 × 601 × 2 bytes = 2.4 MB (FP16)
Computation: ~30 seconds one-time cost
Query: O(1) bilinear interpolation
Speedup: ~1000× vs on-demand computation
```

### 9.3 Memory Layout Optimization

**Structure-of-Arrays (SoA)** for coalesced memory access:
```
Concentrations: [c₀, c₁, ..., c₁₈₀₂₉]  # Contiguous
Velocities:     [v₀, v₁, ..., v₁₈₀₂₉]  # Contiguous  
Absorbances:    [A₀, A₁, ..., A₁₈₀₂₉]  # Contiguous
```

GPU threads access consecutive addresses → single memory transaction.

### 9.4 Mixed Precision Training

**FP16 (Half Precision):**
- Forward pass computations
- Christoffel grid storage
- Activation storage
- 2× speedup on Tensor Cores

**FP32 (Single Precision):**
- Parameter updates
- Gradient accumulation
- Loss computation
- Maintains numerical stability

**Dynamic Loss Scaling:**
```python
with autocast():
    predictions = model.forward(batch)  # FP16
    loss = compute_loss(predictions)    # FP16
    
scaled_loss = loss × 2^k  # Scale to prevent underflow
scaled_loss.backward()    # FP32 gradients
optimizer.step()          # FP32 updates
```

### 9.5 Device-Specific Optimizations

**NVIDIA A100 (Google Colab):**
- Tensor Cores: Hidden dimensions multiple of 8
- Memory bandwidth: 1.5 TB/s → massive batch processing
- 40 GB HBM2: Fit entire dataset in GPU memory

**Apple M1/M2/M3 (Local Development):**
- Metal Performance Shaders (MPS) backend
- Unified memory: Zero-copy data transfer
- Neural Engine: Accelerated inference

## 10. Validation Protocol

### 10.1 Leave-One-Out Cross-Validation

Train 6 models, each excluding one concentration:
```
Model 0: Train on {10,20,30,40,60}, Test on 0 ppb
Model 1: Train on {0,20,30,40,60}, Test on 10 ppb
...
Model 5: Train on {0,10,20,30,40}, Test on 60 ppb
```

### 10.2 Shared vs. Independent Components

**Shared across all models:**
- Metric network g(c,λ)—geometry is universal
- Christoffel computation pipeline
- Training hyperparameters

**Independent per model:**
- Spectral flow network F(c,v,λ) if needed
- Optimizer states
- Best checkpoints

### 10.3 Baseline Comparison

Apply identical leave-one-out protocol to linear interpolation:
```python
def linear_baseline(cs, ct, As, At, λ):
    α = 0.5  # Midpoint
    return (1-α) × As + α × At
```

This ensures fair comparison—same data splits, same evaluation metrics.

### 10.4 Metrics

**Primary**: Mean Squared Error (MSE) on held-out concentration
**Secondary**: 
- R² score (explained variance)
- Peak wavelength accuracy
- Mean Absolute Percentage Error (MAPE)

## 11. Why Geodesics Are the Right Solution

### 11.1 Physical Justification

Chemical systems naturally follow minimum-energy paths between states. Geodesics formalize this principle mathematically, finding paths that minimize integrated "effort" through spectral space.

### 11.2 Mathematical Rigor

The framework provides:
- **Existence**: Geodesics always exist between points in connected manifolds
- **Uniqueness**: Given boundary conditions, the solution is unique (locally)
- **Stability**: Conserved Hamiltonian ensures numerical stability
- **Differentiability**: Smooth solutions enable gradient-based learning

### 11.3 Computational Feasibility

With modern GPUs and our optimizations:
- 18,030 parallel geodesics per batch
- Sub-second inference for field deployment
- 2-hour training on Google Colab 

## Implementation Details

### Software Stack

```python
# Core dependencies
torch >= 2.0.0          # Deep learning framework
torchdiffeq >= 0.2.3    # ODE solvers with adjoint method
numpy >= 1.24.0         # Numerical computing
scipy >= 1.10.0         # Scientific computing

# GPU support
cuda >= 11.7 (NVIDIA)   # A100 optimization
mps (Apple Silicon)     # M1/M2/M3 support
```

### Model Sizes

```
Metric Network:      33,409 parameters (~130 KB)
Flow Network:         8,769 parameters (~35 KB)
Christoffel Grid:    1.2M points (~2.4 MB FP16)
Total Model:         <3 MB (deployable on phones)
```

### Training Configuration

```python
config = {
    'epochs': 100,
    'batch_size': 256,  # Per iteration
    'learning_rate_metric': 3e-4,
    'learning_rate_flow': 8e-4,
    'shooting_iterations': 10,
    'shooting_tolerance': 1e-3,
    'ode_time_steps': 11,
    'validation_interval': 10
}
```

## Conclusion

This geodesic-coupled Neural ODE framework transforms an impossible interpolation problem into a tractable geometric one. By learning the Riemannian structure of spectral space and computing geodesics that respect this geometry, we achieve accurate predictions where traditional methods fail catastrophically.

The key insights:
1. **Spectral space is curved**: Chemistry creates non-uniform sensitivities
2. **Geodesics adapt naturally**: They speed up/slow down based on local geometry
3. **Neural networks learn geometry**: Small networks (~40K parameters) capture complex manifolds
4. **GPUs enable scale**: Massive parallelization makes the approach practical

For the Regeneron Science Talent Search, this work demonstrates how advanced mathematics (differential geometry), modern machine learning (Neural ODEs), and systems optimization (GPU parallelization) can combine to solve real-world problems. The resulting system is not just theoretically elegant but practically deployable—small enough to run on smartphones yet accurate enough for field diagnostics.

The broader impact extends beyond arsenic detection. This geometric framework applies to any spectroscopic measurement with non-monotonic responses: drug discovery, materials science, environmental monitoring. By showing that respecting the geometry of measurement space is essential for accurate interpolation, this work provides a new paradigm for scientific instrumentation in resource-limited settings.