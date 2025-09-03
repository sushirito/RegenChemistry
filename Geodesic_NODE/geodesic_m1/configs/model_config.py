"""Network architecture settings for M1 Mac geodesic NODE"""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class ModelConfig:
    """Neural network architecture configuration"""
    
    # Metric network architecture
    metric_input_dim: int = 2  # [c, λ]
    metric_hidden_dims: List[int] = None
    metric_activation: str = 'tanh'
    metric_use_batch_norm: bool = False
    
    # Spectral flow network architecture  
    flow_input_dim: int = 3  # [c, v, λ]
    flow_hidden_dims: List[int] = None
    flow_activation: str = 'tanh'
    
    # Decoder network architecture
    decoder_input_features: int = 5  # [c_final, c_mean, path_length, max_velocity, wavelength]
    decoder_hidden_dims: List[int] = None
    decoder_activation: str = 'tanh'
    decoder_dropout_rate: float = 0.0
    
    def __post_init__(self):
        """Set default architectures if not provided"""
        if self.metric_hidden_dims is None:
            self.metric_hidden_dims = [128, 256]  # M1-optimized
            
        if self.flow_hidden_dims is None:
            self.flow_hidden_dims = [64, 128]  # M1-optimized
            
        if self.decoder_hidden_dims is None:
            self.decoder_hidden_dims = [64, 128]  # M1-optimized
            
    def get_parameter_count(self) -> Dict[str, int]:
        """Estimate parameter counts for each network"""
        counts = {}
        
        # Metric network
        metric_params = self.metric_input_dim
        for dim in self.metric_hidden_dims:
            metric_params = metric_params * dim + dim  # weights + biases
        metric_params += self.metric_hidden_dims[-1] + 1  # output layer
        counts['metric_network'] = metric_params
        
        # Flow network
        flow_params = self.flow_input_dim
        for dim in self.flow_hidden_dims:
            flow_params = flow_params * dim + dim
        flow_params += self.flow_hidden_dims[-1] + 1
        counts['flow_network'] = flow_params
        
        # Decoder network
        decoder_params = self.decoder_input_features
        for dim in self.decoder_hidden_dims:
            decoder_params = decoder_params * dim + dim
        decoder_params += self.decoder_hidden_dims[-1] + 1
        counts['decoder_network'] = decoder_params
        
        counts['total_estimated'] = sum(counts.values())
        
        return counts