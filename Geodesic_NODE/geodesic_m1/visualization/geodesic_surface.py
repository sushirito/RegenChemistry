"""
Generate continuous geodesic surfaces for visualization
"""

import numpy as np
import torch
from typing import Optional, Tuple


def generate_geodesic_surface(
    model,
    wavelengths: np.ndarray,
    concentrations: list,
    holdout_idx: int,
    c_range: np.ndarray,
    wl_subset: np.ndarray,
    dataset,
    device: torch.device = torch.device('cpu')
) -> np.ndarray:
    """
    Generate full geodesic surface by computing predictions at multiple targets
    
    Args:
        model: Trained geodesic model
        wavelengths: Full wavelength array
        concentrations: All concentration values
        holdout_idx: Index of holdout concentration
        c_range: Concentration values for surface
        wl_subset: Wavelength subset for surface
        dataset: Dataset with normalization parameters
        device: Computation device
        
    Returns:
        Surface array [len(wl_subset), len(c_range)]
    """
    # Get training concentrations (exclude holdout)
    train_concs = [concentrations[i] for i in range(len(concentrations)) if i != holdout_idx]
    
    surface = np.zeros((len(wl_subset), len(c_range)))
    
    model.eval()
    with torch.no_grad():
        for j, wl_val in enumerate(wl_subset):
            # Find closest wavelength index
            wl_idx = np.argmin(np.abs(wavelengths - wl_val))
            
            for k, c_target in enumerate(c_range):
                # Find best source concentration
                # Use nearest training point as source
                distances = [abs(tc - c_target) for tc in train_concs]
                nearest_idx = np.argmin(distances)
                c_source = train_concs[nearest_idx]
                
                # Normalize inputs
                c_source_norm = (c_source - dataset.c_mean) / dataset.c_std
                c_target_norm = (c_target - dataset.c_mean) / dataset.c_std
                wl_norm = (wl_val - dataset.lambda_mean) / dataset.lambda_std
                
                # Create tensors
                c_source_t = torch.tensor([c_source_norm], dtype=torch.float32, device=device)
                c_target_t = torch.tensor([c_target_norm], dtype=torch.float32, device=device)
                wl_t = torch.tensor([wl_norm], dtype=torch.float32, device=device)
                
                # Get prediction
                try:
                    output = model(c_source_t, c_target_t, wl_t)
                    pred = output['absorbance'].cpu().numpy()[0]
                    
                    # Denormalize
                    pred = pred * dataset.A_std + dataset.A_mean
                    surface[j, k] = pred
                except:
                    # If prediction fails, use nearest neighbor
                    surface[j, k] = surface[j, max(0, k-1)]
    
    return surface