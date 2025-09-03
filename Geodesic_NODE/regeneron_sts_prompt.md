# Regeneron STS Technical Paper Writing Instructions

## Context and Objective

You are writing a technical explanation for the Regeneron Science Talent Search (STS), one of the most prestigious high school science competitions. Judges are accomplished scientists who value:
- **Technical rigor** with clear physical intuition
- **Innovation** that solves real problems
- **Justification** for every design choice
- **Impact** on human welfare

Your paper should explain a novel spectral interpolation method using differential geometry to address arsenic detection in drinking water.

## Core Message to Convey

**The Two-Stage Solution:**
1. **Learn the geometry**: Train a neural network to learn the metric tensor g(c,λ) that captures how rapidly spectra change at each point in concentration-wavelength space
2. **Compute optimal paths**: Use this learned metric to compute geodesics - the optimal interpolation paths through spectral space

## Writing Guidelines

### 1. Start with Concrete Failure

**DO NOT** begin with abstract statements about interpolation or geometry.

**DO** begin with a specific example:
"You measure UV-visible spectra at arsenic concentrations of 0, 10, 20, 30, 40, and 60 ppb. You need to predict the spectrum at 45 ppb. Linear interpolation between 40 and 60 ppb should work - but it fails catastrophically. At wavelength 459 nm, the actual absorbance exhibits non-monotonic behavior: it increases from 40 to 45 ppb, then decreases to 60 ppb. Linear interpolation, assuming monotonic change, produces errors exceeding 100%."

### 2. Build Physical Intuition Before Mathematics

**For each concept, follow this progression:**
1. Physical phenomenon (what happens in the real world)
2. Intuitive explanation (why it happens)
3. Mathematical formalization (how we model it)
4. Computational implementation (how we solve it)

**Example for geodesics:**
- Physical: "Spectral responses can loop back on themselves"
- Intuitive: "Like navigating mountains - the shortest path isn't straight"
- Mathematical: "Geodesic equation: d²c/dt² = -Γ(c,λ)(dc/dt)²"
- Computational: "Solve via shooting method with parallel GPU execution"

### 3. Justify Every Technical Decision

**Template for justifications:**
"We chose [DECISION] because [SPECIFIC PROBLEM IT SOLVES]. The alternative [OTHER APPROACH] fails because [CONCRETE REASON]."

**Key justifications to include:**

**Why Differential Geometry:**
"Standard interpolation assumes Euclidean space where straight lines are optimal. When spectral responses are non-monotonic - increasing then decreasing with concentration - no straight line in concentration space can capture this behavior. We need a framework where paths can curve to follow the natural spectral evolution. Differential geometry provides exactly this: we learn a metric that makes non-monotonic responses correspond to curved geodesics."

**Why 1D Manifolds Instead of 2D:**
"We model each wavelength as its own 1D Riemannian manifold over concentration space rather than a single 2D manifold over (concentration, wavelength) space. This is justified because:
- Wavelengths are densely sampled (every 1 nm from 200-800 nm) - interpolation between wavelengths is unnecessary
- Each wavelength exhibits unique chemical behavior with different absorption mechanisms
- 1D geodesics require solving a second-order ODE, while 2D geodesics require solving complex partial differential equations
- The 601 independent 1D problems naturally parallelize on GPU architecture"

**Why One Neural Network Instead of 600:**
"The naive approach would train a separate model for each wavelength. Fatal flaw: with only 6 concentration measurements per wavelength, any model would severely overfit. Instead, we train a single network g(c,λ) that takes wavelength as input:
- Leverages all 3,606 data points (601 wavelengths × 6 concentrations) instead of just 6
- Captures shared chemical patterns across wavelengths while preserving wavelength-specific behavior
- Provides implicit regularization through parameter sharing
- Enables generalization to wavelengths not seen during training"

**Why Coupled ODEs for Spectral Evolution:**
"The geodesic gives us the optimal path through concentration space, but we also need the absorbance values along that path. We couple the geodesic equation with a spectral evolution equation dA/dt = F(c, v, λ). The velocity v provides crucial information - rapid velocity indicates we're traversing a chemical transition region where absorbance changes differently than in stable regions."

**Why Pre-compute Christoffel Symbols:**
"Computing Christoffel symbols Γ = ½g⁻¹(∂g/∂c) requires three neural network forward passes for finite difference approximation. During geodesic integration, we need these values at every integration step (typically 10-50 steps) for every geodesic. Pre-computing on a fine grid (2000 points) and using bilinear interpolation reduces computation by factor of >1000 with <0.1% accuracy loss."

### 4. Explain the Mathematical Framework Clearly

**Structure for mathematical sections:**

1. **State the goal**: What are we trying to compute?
2. **Define the mathematics**: Equations with clear notation
3. **Interpret physically**: What does each term mean?
4. **Connect to implementation**: How do we actually compute this?

**Example for the metric tensor:**
```
Goal: Quantify how "difficult" it is to move through spectral space at each point

Mathematics: 
- Metric tensor: g(c,λ) : ℝ × ℝ → ℝ₊
- Learned via neural network: g = softplus(f_θ(c_norm, λ_norm)) + 0.1
- Ensures positivity while remaining differentiable

Physical interpretation:
- High g(c,λ): Spectra change rapidly here - small concentration changes produce large spectral changes
- Low g(c,λ): Spectra are stable - concentration changes produce predictable spectral shifts
- This captures regions of chemical transitions vs stable regions

Implementation:
- 3-layer neural network with 128-256 hidden units
- Tanh activations for smooth gradients
- Softplus output ensures g > 0 (required for valid metric)
```

### 5. Emphasize Innovation and Impact

**Throughout the paper, highlight:**

1. **Novel contributions:**
   - First application of learned Riemannian geometry to spectral interpolation
   - Coupled ODE system that evolves both position and spectral values
   - Massive parallelization strategy for practical training times
   - Leave-one-out validation with shared metric learning

2. **Real-world impact:**
   - Enables accurate arsenic detection at concentrations not directly measured
   - Makes field deployment practical with rapid training and inference
   - Reduces cost by minimizing required reference measurements
   - Applicable to other spectroscopic measurement problems

3. **Technical achievements:**
   - Orders of magnitude improvement over linear interpolation
   - Successful handling of non-monotonic responses
   - Efficient GPU implementation enabling practical use

### 6. Use Precise but Accessible Language

**Good example:**
"The Christoffel symbol Γ(c,λ) = ½g⁻¹(∂g/∂c) encodes how geodesics curve. When Γ > 0, geodesics bend toward higher concentrations - the geometry 'pulls' paths upward. This occurs in regions where spectral changes accelerate with increasing concentration."

**Bad example:**
"The Christoffel symbol of the first kind determines geodesic curvature in the Riemannian manifold."
(Correct but provides no intuition)

### 7. Structure for Maximum Clarity

**Use consistent section patterns:**
- **The Challenge**: What specific problem are we solving?
- **Why Current Methods Fail**: Concrete examples of failure modes
- **Our Insight**: The key realization that enables our solution
- **Technical Approach**: How we implement the insight
- **Validation**: Proof that it works

### 8. Include Computational Realities

Don't just present the mathematics - explain how you make it practical:

- **Computational complexity**: Number of operations, memory requirements
- **Optimization strategies**: Pre-computation, parallelization, mixed precision
- **Engineering tradeoffs**: Accuracy vs speed, memory vs computation
- **Scaling considerations**: How performance changes with problem size

### 9. Connect Everything to the Core Problem

Every section should answer: "How does this help predict arsenic concentrations?"

Never introduce complexity without explaining why simpler approaches fail.

### 10. Writing Style Requirements

- **Confident but not arrogant**: "Our approach recognizes..." not "We were the first to discover..."
- **Technical but clear**: Define terms on first use, provide intuition
- **Concise but complete**: No unnecessary verbosity, but don't skip crucial steps
- **Problem-focused**: Always connect back to arsenic detection

## Document Structure to Follow

You will receive a detailed technical outline (`sts_technical_outline.md`) with the specific structure. Follow it closely while incorporating these writing guidelines.

## Remember

**Your audience**: Technically sophisticated scientists who don't know your specific problem. They understand advanced mathematics but need to be convinced your approach is both necessary and elegant.

**Your goal**: Demonstrate that differential geometry isn't unnecessary complexity - it's the natural and necessary framework for this problem.

**Your impact**: Show how mathematical innovation directly addresses a humanitarian crisis affecting millions.

## Final Checklist

Before completing each section, verify:
- [ ] Concrete example provided before abstraction
- [ ] Physical intuition given for mathematical concepts
- [ ] Every design choice explicitly justified
- [ ] Technical terms defined on first use
- [ ] Connection to arsenic detection made clear
- [ ] Both innovation and impact emphasized

Write as if the reader's understanding and appreciation of your work will determine whether this life-saving technology gets deployed.