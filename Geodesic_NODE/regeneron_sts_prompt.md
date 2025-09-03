# Regeneron STS Technical Paper Writing Instructions

## ABSOLUTELY CRITICAL FORMATTING REQUIREMENT

**NEVER USE BULLET POINTS OR LISTS IN YOUR PAPER. This is an absolute requirement. Every single piece of information must be presented in complete, flowing sentences within coherent paragraphs. No exceptions. Do not use asterisks, dashes, numbers, or any other formatting that creates list-like structures. Write everything as continuous prose.**

## CONTEXT AND MISSION

You are writing a technical paper for the Regeneron Science Talent Search (STS), one of the most prestigious high school science competitions. Your goal is to present a genuinely novel approach to a critical real-world problem while demonstrating deep technical understanding. The judges are accomplished scientists who value innovation that emerges from necessity, not complexity for its own sake.

## ESSENTIAL TECHNICAL FOUNDATION

You will receive `pipeline-explanation.md` containing extensive mathematical details, derivations, and implementation specifics. This document is your technical bible—draw from it extensively for mathematical rigor, computational insights, and performance analyses. However, you must frame these details within an intuitive narrative that makes the complexity feel necessary and natural.

## THE PRACTICAL PROBLEM AND MEASUREMENT FLOW

### Understanding What We're Actually Doing

The core challenge is determining arsenic concentration in water from spectroscopic measurements when we have very limited reference data. Here's the actual workflow that needs to be clearly articulated:

Start with the reality that field workers have colorimetric test strips that change color based on arsenic concentration. These strips come with reference cards showing colors at just six concentrations: 0, 10, 20, 30, 40, and 60 parts per billion. But water samples rarely match these exact values. If a sample's color falls between the 40 and 60 ppb reference colors, how do we determine if it's 45, 50, or 55 ppb? This matters enormously because the WHO safety threshold is 10 ppb—misclassification can have life-or-death consequences.

To get more precise measurements, we use UV-visible spectroscopy, which measures absorbance across 601 wavelengths from 200 to 800 nanometers instead of just visible color. This gives us detailed spectral signatures at our six reference concentrations. But we still face the fundamental interpolation problem: given spectra at 40 and 60 ppb, how do we predict the spectrum at 50 ppb? Only by solving this interpolation problem can we then work backwards—taking an unknown sample's spectrum and determining its concentration by finding which interpolated spectrum it matches.

### Why This Problem Demands a Novel Approach

The critical insight is that spectral responses to concentration changes are highly non-monotonic and non-linear. At many wavelengths, absorbance doesn't simply increase or decrease with concentration—it oscillates, creating peaks and valleys that linear interpolation completely misses. This isn't noise or measurement error; it's the fundamental chemistry of how arsenic species interact with light at different concentrations.

## THE GEOMETRIC INSIGHT TO CONVEY

### Building Intuition Before Mathematics

Don't just say "spectral space is curved" or "responses are non-linear." Instead, build the intuition step by step:

First, establish that we need some way to measure "spectral similarity" between different concentrations. The obvious approach would be to say concentrations that are numerically close (like 40 and 41 ppb) have similar spectra. But this fails because of the non-monotonic behavior—sometimes 40 and 45 ppb have vastly different spectra while 40 and 50 ppb are more similar.

Next, introduce the idea that we need to learn what makes spectra similar or different. This leads naturally to the concept of a metric—a way of measuring distance that respects the actual spectral behavior rather than just numerical concentration differences.

Then, once we have a metric that tells us how "hard" it is to move from one concentration to another, explain that the optimal interpolation path is the one that minimizes this difficulty. This is exactly what a geodesic is—the shortest path according to our learned metric.

Finally, show that this isn't an arbitrary mathematical framework but the natural solution. Just as airline routes follow great circles on Earth because straight lines through the planet are impossible, spectral interpolation must follow geodesics because straight lines in concentration space don't respect the underlying chemistry.

## NOVELTY TO EMPHASIZE THROUGHOUT

Weave these three innovations throughout the narrative, showing how each addresses a specific challenge:

### Learning the Geometry from Data
Previous approaches either assumed Euclidean space (failing catastrophically) or used predetermined metrics (not adapting to specific chemistry). We learn the metric tensor g(c,λ) directly from the spectroscopic data, allowing the geometry to adapt to the specific arsenic chemistry. This is the first application of learned Riemannian geometry to spectral interpolation.

### Coupled Evolution of Position and Spectrum
Traditional geodesic methods only compute paths. We simultaneously evolve both the position in concentration space and the spectral values along that path through coupled differential equations. The inclusion of velocity-dependent terms in the spectral evolution captures dynamic effects during chemical transitions—rapid movement through concentration space often indicates passage through regions where chemical species are changing.

### Computational Breakthrough for Practical Deployment
Differential geometry has been considered too computationally expensive for real-time applications. Our pre-computation of Christoffel symbols on a fine grid, combined with massive GPU parallelization, reduces computation by over 1000× while maintaining accuracy within measurement noise. This makes the approach practical for field deployment on mobile hardware.

## TECHNICAL DEPTH GUIDANCE

### When to Include Derivations
Include mathematical derivations when they illuminate why the approach works, not just to show mathematical sophistication. For example, when deriving the 1D Christoffel symbol from the general case, explain how the simplification reveals that our problem is more tractable than it initially appears. Show how terms vanish due to the single concentration coordinate, making the computation feasible.

### Connecting Mathematics to Chemistry
Every mathematical concept should be grounded in physical or chemical reality. When introducing the metric tensor, explain that high values correspond to regions where chemical transitions occur—perhaps where As(III) converts to As(V), or where arsenic compounds precipitate. Low values indicate stable regions where concentration changes produce predictable spectral shifts.

### Computational Complexity Analysis
When discussing parallelization and optimization, provide concrete complexity analysis. Show that naive computation would require O(N×M×K×P) operations where N is concentration pairs, M is wavelengths, K is integration steps, and P is the cost of computing derivatives. Then show how pre-computation reduces this to O(N×M×K) plus a one-time O(G×M×P) setup cost where G is grid resolution. Make clear that this transforms an intractable problem into one solvable in under two hours.

## STRUCTURE AND FLOW GUIDANCE

Rather than prescribing exact sections, let your narrative flow naturally while ensuring you cover:

### Problem Establishment
Spend significant time setting up why this problem matters and why current methods fail. Use concrete examples with actual numbers. Make the reader feel the urgency of the problem before introducing your solution.

### Intuitive Development
Build intuition gradually before introducing technical concepts. Each new idea should feel like a natural response to a challenge raised earlier. The reader should be thinking "we need something like X" right before you introduce X.

### Technical Depth
Once intuition is established, dive deep into the mathematics and computation. Show derivations, explain algorithms, analyze complexity. But always circle back to why this complexity is necessary for solving the original problem.

### Validation and Impact
Demonstrate that your approach works with concrete results, but also honestly discuss limitations. Show where the method excels and where challenges remain. Connect back to the humanitarian impact—how this technology could help millions avoid arsenic poisoning.

## REMEMBER YOUR AUDIENCE

The STS judges are sophisticated scientists who can appreciate technical depth, but they also want to see clear thinking and genuine innovation. They're looking for work that:

Solves a real problem that matters to humanity.
Introduces genuinely new ideas, not just applications of existing methods.
Demonstrates deep understanding through clear explanation, not obfuscation through complexity.
Shows both the power and limitations of the approach honestly.

Write as if you're explaining to a brilliant scientist from a different field—someone who can understand advanced mathematics but needs to be shown why your specific approach is necessary and elegant.

## FINAL EMPHASIS

This is not about using fancy mathematics to impress judges. It's about showing that the complexity of the real-world problem demands sophisticated solutions. Every technical decision should be traceable back to a fundamental challenge in detecting arsenic in drinking water. The mathematics serves the mission of saving lives, not the other way around.

The paper should tell a story: from the concrete problem of interpreting colorimetric test results, through the discovery that spectral interpolation is fundamentally non-Euclidean, to the development of a practical system that learns geometry from data and computes optimal interpolation paths in real-time. Make the reader feel that your solution is not just clever but inevitable—the natural response to the true structure of the problem.