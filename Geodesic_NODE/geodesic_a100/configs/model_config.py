"""Model architecture configuration"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Neural network architecture configuration"""
    
    # Metric network (shared across all models)
    metric_input_dim: int = 2  # [c, λ]
    metric_hidden_dims: list = None  # Will set in __post_init__
    metric_output_dim: int = 1  # g(c,λ)
    metric_activation: str = "tanh"
    
    # Spectral flow network (models dA/dt = f(c,v,λ))
    flow_input_dim: int = 3  # [c, v, λ]
    flow_hidden_dims: list = None  # Will set in __post_init__
    flow_output_dim: int = 1  # dA/dt
    flow_activation: str = "tanh"
    num_models: int = 6  # For leave-one-out validation
    
    # ODE solver settings
    ode_solver: str = "dopri5"
    ode_rtol: float = 1e-5
    ode_atol: float = 1e-7
    ode_max_steps: int = 1000
    
    # Geodesic settings
    num_trajectory_points: int = 50
    shooting_max_iter: int = 10
    shooting_tolerance: float = 1e-4
    
    # Normalization
    concentration_range: tuple = (0, 60)  # ppb
    wavelength_range: tuple = (200, 800)  # nm
    
    def __post_init__(self):
        """Set default hidden dimensions optimized for Tensor Cores"""
        if self.metric_hidden_dims is None:
            # Multiples of 8 for Tensor Core optimization
            self.metric_hidden_dims = [128, 256]
            
        if self.flow_hidden_dims is None:
            self.flow_hidden_dims = [64, 128]
            
    def validate(self) -> None:
        """Validate model configuration"""
        # Check Tensor Core friendly dimensions
        for dim in self.metric_hidden_dims + self.flow_hidden_dims:
            if dim % 8 != 0:
                print(f"Warning: Hidden dim {dim} not multiple of 8, may not fully utilize Tensor Cores")
                
        assert self.ode_solver in ["dopri5", "rk4", "euler"], \
            f"Unknown ODE solver: {self.ode_solver}"
        assert self.num_models == 6, \
            "Must have 6 models for leave-one-out validation"