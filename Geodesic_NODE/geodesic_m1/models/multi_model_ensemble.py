"""
Multi-model ensemble for leave-one-out validation
Each model trained with one concentration excluded
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
from .spectral_decoder import SpectralDecoder


class MultiModelEnsemble(nn.Module):
    """Ensemble of spectral decoders for leave-one-out validation"""
    
    def __init__(self,
                 n_concentrations: int = 6,
                 input_features: int = 5,
                 hidden_dims: List[int] = None,
                 activation: str = 'tanh',
                 dropout_rate: float = 0.0):
        """
        Initialize multi-model ensemble
        
        Args:
            n_concentrations: Number of concentration levels (6 models)
            input_features: Number of input features per model
            hidden_dims: Hidden layer dimensions for each decoder
            activation: Activation function
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        self.n_concentrations = n_concentrations
        self.input_features = input_features
        
        # Create one decoder for each leave-one-out model
        self.decoders = nn.ModuleList([
            SpectralDecoder(
                input_features=input_features,
                hidden_dims=hidden_dims,
                activation=activation,
                dropout_rate=dropout_rate
            )
            for _ in range(n_concentrations)
        ])
        
        # Store which concentration each model excludes
        self.excluded_concentrations = list(range(n_concentrations))
        
    def forward(self, features: torch.Tensor, 
                model_idx: int) -> torch.Tensor:
        """
        Forward pass through specific decoder model
        
        Args:
            features: Input features [batch_size, input_features]
            model_idx: Which decoder model to use (0-5)
            
        Returns:
            absorbance: Predicted absorbance [batch_size, 1]
        """
        if model_idx < 0 or model_idx >= self.n_concentrations:
            raise ValueError(f"Model index {model_idx} out of range [0, {self.n_concentrations-1}]")
            
        return self.decoders[model_idx](features)
        
    def forward_all(self, features: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Forward pass through all decoder models
        
        Args:
            features: Input features [batch_size, input_features]
            
        Returns:
            Dictionary mapping model_idx -> predictions
        """
        predictions = {}
        for model_idx in range(self.n_concentrations):
            predictions[model_idx] = self.decoders[model_idx](features)
        return predictions
        
    def predict_with_ensemble(self, features: torch.Tensor, 
                             weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Ensemble prediction using all models
        
        Args:
            features: Input features [batch_size, input_features]
            weights: Optional weights for each model [n_concentrations]
            
        Returns:
            ensemble_prediction: Weighted average prediction [batch_size, 1]
        """
        # Get predictions from all models
        all_predictions = self.forward_all(features)
        
        # Stack predictions
        stacked_preds = torch.stack([
            all_predictions[i] for i in range(self.n_concentrations)
        ], dim=-1)  # [batch_size, 1, n_concentrations]
        
        # Apply weights (uniform if not provided)
        if weights is None:
            weights = torch.ones(self.n_concentrations, device=features.device) / self.n_concentrations
        else:
            weights = weights / weights.sum()  # Normalize
            
        # Weighted average
        ensemble_pred = (stacked_preds * weights).sum(dim=-1)  # [batch_size, 1]
        
        return ensemble_pred
        
    def get_model_for_validation(self, excluded_concentration_idx: int) -> nn.Module:
        """
        Get the specific model that excludes given concentration
        
        Args:
            excluded_concentration_idx: Index of concentration to exclude (0-5)
            
        Returns:
            Corresponding decoder model
        """
        if excluded_concentration_idx not in self.excluded_concentrations:
            raise ValueError(f"No model excludes concentration {excluded_concentration_idx}")
            
        return self.decoders[excluded_concentration_idx]
        
    def compute_validation_loss(self, features_dict: Dict[int, torch.Tensor],
                               targets_dict: Dict[int, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute validation loss for all leave-one-out models
        
        Args:
            features_dict: Dictionary mapping excluded_idx -> features
            targets_dict: Dictionary mapping excluded_idx -> target absorbances
            
        Returns:
            Dictionary with individual and total losses
        """
        losses = {}
        total_loss = 0.0
        
        for excluded_idx in range(self.n_concentrations):
            if excluded_idx in features_dict and excluded_idx in targets_dict:
                # Get predictions from corresponding model
                predictions = self.forward(features_dict[excluded_idx], excluded_idx)
                targets = targets_dict[excluded_idx]
                
                # Compute MSE loss
                model_loss = nn.functional.mse_loss(predictions, targets)
                losses[f'model_{excluded_idx}_loss'] = model_loss
                total_loss += model_loss
                
        losses['total_loss'] = total_loss / len(features_dict)
        return losses
        
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the ensemble"""
        total_params = sum(p.numel() for p in self.parameters())
        params_per_model = total_params // self.n_concentrations
        
        return {
            'n_models': self.n_concentrations,
            'total_parameters': total_params,
            'parameters_per_model': params_per_model,
            'excluded_concentrations': self.excluded_concentrations,
            'input_features': self.input_features
        }