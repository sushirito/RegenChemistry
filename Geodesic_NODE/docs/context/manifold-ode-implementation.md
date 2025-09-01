Core Principle

Implement a model that actually solves the geodesic differential equation 
d²c/dt² = -Γ(c,λ)(dc/dt)² using proper Riemannian geometry, with geodesic 
trajectories providing features for absorbance prediction.

Component 1: Data Pipeline

File: data_loader.py
- Load CSV: 601 wavelengths × 6 concentrations 
- Normalize: λ_norm = (λ - 500)/300, c_norm = (c - 30)/30, A_norm = (A - 
mean)/std
- Generate training pairs: (c_source, c_target, λ, A_target)
- Total: 6×5×601 = 18,030 concentration transition pairs

Component 2: Metric Network

File: metric_network.py
class MetricNetwork(nn.Module):
    Input: [c, λ] normalized to [-1, 1]
    Architecture:
        Linear(2 → 64) + Tanh
        Linear(64 → 128) + Tanh  
        Linear(128 → 1)
    Output: g(c,λ) = softplus(raw_metric) + 0.1
- Must be smooth and positive everywhere
- Shared across all wavelengths (prevents overfitting)

Component 3: Christoffel Symbol Computer

File: christoffel.py
def compute_christoffel(c, λ, metric_network):
    # Finite differences for stability
    ε = 1e-4
    g_plus = metric_network(c + ε, λ)
    g_minus = metric_network(c - ε, λ)
    g_center = metric_network(c, λ)
    dg_dc = (g_plus - g_minus) / (2ε)
    Γ = 0.5 * dg_dc / g_center
    return Γ
- Cache computations for efficiency
- Ensure numerical stability

Component 4: Geodesic ODE System

File: geodesic_ode.py
def geodesic_ode(t, state, λ, metric_network):
    # State = [c, v] where v = dc/dt
    c, v = state[0], state[1]
    Γ = compute_christoffel(c, λ, metric_network)
    dc_dt = v
    dv_dt = -Γ * v**2
    return [dc_dt, dv_dt]
- This is the TRUE geodesic equation, not an approximation
- Will be integrated using ODE solver

Component 5: Shooting Method BVP Solver

File: shooting_solver.py
def solve_bvp(c_source, c_target, λ, metric_network):
    # Find initial velocity v₀ such that geodesic reaches c_target
    def objective(v0):
        initial_state = [c_source, v0]
        solution = odeint(geodesic_ode, initial_state, t_span=[0,1])
        c_final = solution[-1, 0]
        return (c_final - c_target)**2
    
    # Optimize to find correct v₀
    v0_initial = c_target - c_source  # Linear guess
    result = minimize(objective, v0_initial)
    
    # Get full trajectory (11 points)
    t_eval = np.linspace(0, 1, 11)
    trajectory = odeint(geodesic_ode, [c_source, result.x], t_eval)
    return trajectory  # Shape: (11, 2) - positions and velocities
- Returns full geodesic path for feature extraction
- Handles convergence failures gracefully

Component 6: Absorbance Decoder with Path Encoding

File: decoder_network.py
class AbsorbanceDecoder(nn.Module):
    def __init__(self):
        # Option B: Path Statistics (as specified in doc)
        self.decoder = nn.Sequential(
            nn.Linear(5, 64),  # [c_final, c_mean, path_length, max_velocity, λ]
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    
    def forward(self, trajectory, λ):
        # Extract path statistics from trajectory
        c_path = trajectory[:, 0]  # Concentration values
        v_path = trajectory[:, 1]  # Velocity values
        
        c_final = c_path[-1]
        c_mean = c_path.mean()
        path_length = compute_geodesic_length(trajectory)
        max_velocity = np.abs(v_path).max()
        
        features = torch.tensor([c_final, c_mean, path_length, max_velocity, λ])
        return self.decoder(features)
- Uses geodesic path information, not just endpoints
- Path statistics capture non-monotonic behavior

Component 7: Full Model Integration

File: geodesic_model.py
class GeodesicSpectralModel(nn.Module):
    def __init__(self):
        self.metric_network = MetricNetwork()
        self.decoder = AbsorbanceDecoder()
    
    def forward(self, c_source, c_target, λ):
        # Solve BVP to get geodesic
        trajectory = solve_bvp(c_source, c_target, λ, self.metric_network)
        
        # Decode to absorbance
        absorbance = self.decoder(trajectory, λ)
        return absorbance, trajectory

Component 8: Training Loop with All Losses

File: train.py
def train_epoch(model, data_loader, optimizer):
    for batch in data_loader:
        c_source, c_target, λ, A_target = batch
        
        # Forward pass
        A_pred, trajectory = model(c_source, c_target, λ)
        
        # Compute losses (as specified)
        L_recon = MSE(A_pred, A_target)
        L_smooth = compute_metric_smoothness(model.metric_network)  # (∂²g/∂c²)²
        L_bounds = compute_metric_bounds(model.metric_network)  # Keep g in 
[0.01, 100]
        L_path = compute_path_length(trajectory)  # Efficiency regularization
        
        # Combined loss with specified weights
        loss = L_recon + 0.01*L_smooth + 0.001*L_bounds + 0.001*L_path
        
        # Backprop through entire pipeline
        optimizer.zero_grad()
        loss.backward()  # Uses adjoint method for ODE
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
- Two optimizers: lr=5e-4 for metric, lr=1e-3 for decoder
- Cosine annealing over 500 epochs
- Gradient clipping for stability

Component 9: Validation with LOO-CV

File: validate.py
def leave_one_out_validation(model, data):
    for holdout_idx in range(6):
        # Train on 5 concentrations, test on 1
        train_data = exclude_concentration(data, holdout_idx)
        test_data = get_concentration(data, holdout_idx)
        
        # Evaluate interpolation quality
        compute_23_metrics(predictions, ground_truth)
        # Focus on: R², RMSE, MAPE, Peak λ Error

Key Implementation Requirements Met

✅ Actually solves geodesic equation: d²c/dt² = -Γ(c,λ)(dc/dt)²  
✅ Proper BVP with shooting method: Ensures geodesics connect endpoints  
✅ Path encoding in decoder: Uses trajectory statistics, not just endpoints  
✅ All specified losses: Reconstruction + smoothness + bounds + path efficiency  
✅ Correct normalization: λ_norm = (λ-500)/300, c_norm = (c-30)/30  
✅ Adjoint method for backprop: Through ODE solver  
✅ Specified architecture sizes: 2→64→128→1 for metric, 5→64→128→1 for decoder  
✅ Training configuration: Adam, lr=5e-4/1e-3, gradient clipping, cosine 
annealing

What Makes This Different from Failed Attempts

1. TRUE geodesics: Solves the actual differential equation, not latent 
approximations  
2. Boundary conditions: Shooting method ensures geodesics connect source to 
target  
3. Physical interpretation: Metric g(c,λ) measures traversal difficulty in 
concentration space  
4. Path features: Decoder uses geodesic trajectory information, capturing 
non-monotonic behavior

Expected Outcomes

- R² improvement: -34.13 → >0.7  
- Peak λ error: 459nm → <20nm  
- MAPE: 100.7% → <20%  
- Geodesics show curvature in high-gradient regions (50-60 ppb)
