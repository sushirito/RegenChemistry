# Colab Notebook Modification Instructions

## Overview
This document provides exact instructions for modifying `geodesic_a100_colab.ipynb` to implement 6-model leave-one-out validation with comprehensive analysis. The notebook currently has 29 cells and needs specific modifications to avoid corruption.

## Critical Requirements
- **NEVER use Python scripts** (`python -c`) to modify cells
- **ALWAYS use proper notebook editing commands** (NotebookEdit)
- **Preserve existing cell structure** - don't add duplicate training sections
- **Test each modification** before proceeding to the next

## Required Files/Imports
The following files contain necessary code:
- `src/analysis/spectral_validation_metrics.py` - 20+ metrics functions
- Current notebook has all geodesic model code already implemented

## Modification Plan

### 1. Identify Training Cell to Modify
**Current State**: The notebook has a training section that trains a single model
**Target**: Find the cell containing the main training loop and modify it to train 6 models

**Search Pattern**: Look for cell containing:
```python
# Train the model
for epoch in range(num_epochs):
```

### 2. Replace Single-Model Training (Modify Existing Cell)

**Cell to Modify**: The training cell (likely around cell 15-20)

**Replace the existing training section with**:

```python
# 6-Model Leave-One-Out Training
# Train separate models for each concentration holdout

concentrations = [0, 10, 20, 30, 40, 60]  # ppb
trained_models = {}
training_results = {}

print("Starting 6-Model Leave-One-Out Training...")
print("=" * 60)

for holdout_idx, holdout_conc in enumerate(concentrations):
    print(f"\nTraining Model {holdout_idx + 1}/6 - Holdout: {holdout_conc} ppb")
    print("-" * 40)
    
    # Create fresh model for this holdout
    model = GeodesicSpectralModel(
        input_dim=2,
        hidden_dims=[128, 256],
        n_trajectory_points=11,
        shooting_tolerance=1e-4,
        shooting_max_iter=50,
        device=device
    )
    
    # Create training data excluding holdout concentration
    train_concentrations = [c for i, c in enumerate(concentrations) if i != holdout_idx]
    
    # Generate training pairs for this model
    train_pairs = []
    for c_source in train_concentrations:
        for c_target in train_concentrations:
            if c_source != c_target:
                for wavelength in wavelengths:
                    train_pairs.append((c_source, c_target, wavelength))
    
    print(f"Training pairs: {len(train_pairs)}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # Training loop
    model.train()
    epoch_losses = []
    
    for epoch in range(50):  # Reduced epochs for Colab
        total_loss = 0
        successful_geodesics = 0
        total_attempts = 0
        
        # Random sample for efficiency
        sampled_pairs = random.sample(train_pairs, min(1000, len(train_pairs)))
        
        for c_source, c_target, wavelength in sampled_pairs:
            try:
                # Normalize inputs
                c_source_norm = (c_source - 30) / 30
                c_target_norm = (c_target - 30) / 30
                wavelength_norm = (wavelength - 500) / 300
                
                # Get target absorbance
                c_source_idx = concentrations.index(c_source) if c_source in concentrations else None
                c_target_idx = concentrations.index(c_target) if c_target in concentrations else None
                wl_idx = np.argmin(np.abs(wavelengths - wavelength))
                
                if c_target_idx is not None:
                    target_abs = absorbance_matrix[wl_idx, c_target_idx]
                    
                    # Forward pass
                    result = model(
                        torch.tensor([c_source_norm], dtype=torch.float32, device=device),
                        torch.tensor([c_target_norm], dtype=torch.float32, device=device),
                        torch.tensor([wavelength_norm], dtype=torch.float32, device=device)
                    )
                    
                    if result['success']:
                        # Reconstruction loss
                        pred_abs = result['absorbance'].squeeze()
                        target_abs_tensor = torch.tensor([target_abs], dtype=torch.float32, device=device)
                        loss = F.mse_loss(pred_abs, target_abs_tensor)
                        
                        # Regularization
                        smoothness_loss = model.metric_network.get_smoothness_loss(
                            torch.tensor([c_source_norm], dtype=torch.float32, device=device),
                            torch.tensor([wavelength_norm], dtype=torch.float32, device=device)
                        )
                        bounds_loss = model.metric_network.get_bounds_loss(
                            torch.tensor([c_source_norm], dtype=torch.float32, device=device),
                            torch.tensor([wavelength_norm], dtype=torch.float32, device=device)
                        )
                        
                        total_loss_sample = loss + 0.01 * smoothness_loss + 0.001 * bounds_loss
                        
                        # Backward pass
                        optimizer.zero_grad()
                        total_loss_sample.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                        total_loss += total_loss_sample.item()
                        successful_geodesics += 1
                        
                    total_attempts += 1
                    
            except Exception as e:
                continue  # Skip failed geodesics
        
        # Update learning rate
        scheduler.step()
        
        # Record epoch results
        avg_loss = total_loss / max(successful_geodesics, 1)
        success_rate = successful_geodesics / max(total_attempts, 1) * 100
        epoch_losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:2d}: Loss={avg_loss:.4f}, Success={success_rate:.1f}%, LR={scheduler.get_last_lr()[0]:.2e}")
    
    # Store trained model and results
    trained_models[holdout_conc] = {
        'model': model,
        'losses': epoch_losses,
        'final_loss': epoch_losses[-1] if epoch_losses else float('inf')
    }
    
    training_results[holdout_conc] = {
        'final_loss': epoch_losses[-1] if epoch_losses else float('inf'),
        'success_rate': success_rate
    }
    
    print(f"  Model {holdout_idx + 1} Complete - Final Loss: {epoch_losses[-1]:.4f}")

print("\n" + "=" * 60)
print("6-Model Training Complete!")
for conc, results in training_results.items():
    print(f"  {conc} ppb holdout: Loss={results['final_loss']:.4f}, Success={results['success_rate']:.1f}%")
```

### 3. Add Comprehensive Metrics Analysis Cell (New Cell)

**Add this as a new cell after the training cell**:

```python
# Comprehensive 20+ Metrics Analysis
# Compare Basic Interpolation vs Geodesic Interpolation across all 6 holdouts

import sys
sys.path.append('/content/drive/MyDrive/Colab Notebooks/Geodesic_NODE/src/analysis')
from spectral_validation_metrics import calculate_all_metrics, interpolate_holdout_concentration
import pandas as pd

print("Running Comprehensive Spectral Analysis...")
print("=" * 70)

# Filter to 200-800 nm range
wavelength_mask = (wavelengths >= 200) & (wavelengths <= 800)
wl_filtered = wavelengths[wavelength_mask]
abs_filtered = absorbance_matrix[wavelength_mask, :]

# Results storage
all_results = []

for holdout_idx, holdout_conc in enumerate(concentrations):
    print(f"\nAnalyzing Holdout {holdout_idx + 1}/6: {holdout_conc} ppb")
    print("-" * 50)
    
    # Get ground truth spectrum for holdout concentration
    ground_truth = abs_filtered[:, holdout_idx]
    
    # 1. Basic Interpolation
    basic_interpolated = interpolate_holdout_concentration(
        wl_filtered, concentrations, abs_filtered, holdout_idx
    )
    
    # 2. Geodesic Interpolation
    model = trained_models[holdout_conc]['model']
    model.eval()
    
    geodesic_predictions = np.zeros(len(wl_filtered))
    successful_predictions = 0
    
    with torch.no_grad():
        for wl_idx, wavelength in enumerate(wl_filtered):
            try:
                # Use closest available concentration as source
                available_concs = [c for c in concentrations if c != holdout_conc]
                best_source = min(available_concs, key=lambda x: abs(x - holdout_conc))
                
                # Normalize
                c_source_norm = (best_source - 30) / 30
                c_target_norm = (holdout_conc - 30) / 30
                wavelength_norm = (wavelength - 500) / 300
                
                # Predict
                result = model(
                    torch.tensor([c_source_norm], dtype=torch.float32, device=device),
                    torch.tensor([c_target_norm], dtype=torch.float32, device=device),
                    torch.tensor([wavelength_norm], dtype=torch.float32, device=device)
                )
                
                if result['success']:
                    geodesic_predictions[wl_idx] = result['absorbance'].cpu().numpy()
                    successful_predictions += 1
                else:
                    # Fallback to basic interpolation
                    geodesic_predictions[wl_idx] = basic_interpolated[wl_idx]
                    
            except Exception as e:
                geodesic_predictions[wl_idx] = basic_interpolated[wl_idx]
    
    success_rate = successful_predictions / len(wl_filtered) * 100
    print(f"  Geodesic Success Rate: {success_rate:.1f}%")
    
    # Calculate all metrics
    basic_metrics = calculate_all_metrics(ground_truth, basic_interpolated, wl_filtered)
    geodesic_metrics = calculate_all_metrics(ground_truth, geodesic_predictions, wl_filtered)
    
    # Store results
    for metric_name in basic_metrics.keys():
        all_results.append({
            'Holdout_Concentration': holdout_conc,
            'Method': 'Basic_Interpolation',
            'Metric': metric_name,
            'Value': basic_metrics[metric_name]
        })
        
        all_results.append({
            'Holdout_Concentration': holdout_conc,
            'Method': 'Geodesic',
            'Metric': metric_name,
            'Value': geodesic_metrics[metric_name]
        })

# Create comprehensive results DataFrame
results_df = pd.DataFrame(all_results)

# Pivot table for better visualization
comparison_table = results_df.pivot_table(
    index=['Metric'], 
    columns=['Method', 'Holdout_Concentration'], 
    values='Value', 
    aggfunc='first'
).round(4)

print("\n" + "=" * 70)
print("COMPREHENSIVE SPECTRAL ANALYSIS RESULTS")
print("=" * 70)

# Display key metrics summary
key_metrics = ['RMSE', 'R2_Score', 'MAPE', 'SSIM', 'Peak_Wavelength_Error']
print("\nKEY METRICS SUMMARY:")
print("-" * 30)

for metric in key_metrics:
    if metric in comparison_table.index:
        print(f"\n{metric}:")
        metric_data = comparison_table.loc[metric]
        
        # Calculate averages
        basic_avg = metric_data['Basic_Interpolation'].mean()
        geodesic_avg = metric_data['Geodesic'].mean()
        
        print(f"  Basic Interpolation Average:  {basic_avg:.4f}")
        print(f"  Geodesic Average:            {geodesic_avg:.4f}")
        
        # Improvement calculation
        if metric in ['RMSE', 'MAPE', 'Peak_Wavelength_Error']:  # Lower is better
            improvement = (basic_avg - geodesic_avg) / basic_avg * 100
        else:  # Higher is better
            improvement = (geodesic_avg - basic_avg) / abs(basic_avg) * 100
        
        print(f"  Improvement:                 {improvement:+.1f}%")

# Full detailed table
print(f"\n\nFULL RESULTS TABLE ({len(comparison_table)} metrics):")
print("-" * 50)
print(comparison_table.to_string())

# Save results
results_df.to_csv('/content/drive/MyDrive/comprehensive_spectral_analysis.csv', index=False)
comparison_table.to_csv('/content/drive/MyDrive/comparison_table.csv')

print(f"\nResults saved to:")
print(f"  - comprehensive_spectral_analysis.csv")
print(f"  - comparison_table.csv")
```

### 4. Add 6-Panel Visualization Cell (New Cell)

**Add this as a new cell after the metrics analysis**:

```python
# 6-Panel Minimal Visualization
# Ground Truth vs Basic Interpolation vs Geodesic Prediction

import matplotlib.pyplot as plt

# Create 2x3 subplot grid
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Colors
colors = {
    'ground_truth': '#000000',      # Black
    'basic': '#FF4444',             # Red  
    'geodesic': '#0066CC'          # Blue
}

for holdout_idx, holdout_conc in enumerate(concentrations):
    ax = axes[holdout_idx]
    
    # Get data (200-800 nm range)
    ground_truth = abs_filtered[:, holdout_idx]
    
    # Basic interpolation
    basic_interpolated = interpolate_holdout_concentration(
        wl_filtered, concentrations, abs_filtered, holdout_idx
    )
    
    # Geodesic prediction (use pre-computed from analysis)
    model = trained_models[holdout_conc]['model']
    model.eval()
    
    geodesic_predictions = np.zeros(len(wl_filtered))
    
    with torch.no_grad():
        for wl_idx, wavelength in enumerate(wl_filtered):
            try:
                available_concs = [c for c in concentrations if c != holdout_conc]
                best_source = min(available_concs, key=lambda x: abs(x - holdout_conc))
                
                c_source_norm = (best_source - 30) / 30
                c_target_norm = (holdout_conc - 30) / 30
                wavelength_norm = (wavelength - 500) / 300
                
                result = model(
                    torch.tensor([c_source_norm], dtype=torch.float32, device=device),
                    torch.tensor([c_target_norm], dtype=torch.float32, device=device),
                    torch.tensor([wavelength_norm], dtype=torch.float32, device=device)
                )
                
                if result['success']:
                    geodesic_predictions[wl_idx] = result['absorbance'].cpu().numpy()
                else:
                    geodesic_predictions[wl_idx] = basic_interpolated[wl_idx]
                    
            except:
                geodesic_predictions[wl_idx] = basic_interpolated[wl_idx]
    
    # Plot three curves
    ax.plot(wl_filtered, ground_truth, color=colors['ground_truth'], linewidth=2, label='Ground Truth')
    ax.plot(wl_filtered, basic_interpolated, color=colors['basic'], linewidth=1.5, label='Basic')
    ax.plot(wl_filtered, geodesic_predictions, color=colors['geodesic'], linewidth=1.5, label='Geodesic')
    
    # Minimal styling
    ax.set_title(f'{holdout_conc} ppb', fontsize=12, fontweight='bold')
    ax.set_xlim(200, 800)
    ax.grid(True, alpha=0.3)
    
    # Only label outer plots
    if holdout_idx >= 3:  # Bottom row
        ax.set_xlabel('Wavelength (nm)')
    if holdout_idx % 3 == 0:  # Left column
        ax.set_ylabel('Absorbance')
    
    # Legend only on first plot
    if holdout_idx == 0:
        ax.legend(fontsize=10)

plt.tight_layout()
plt.suptitle('Spectral Interpolation Comparison (200-800 nm)', fontsize=14, fontweight='bold', y=1.02)
plt.show()

# Save figure
plt.savefig('/content/drive/MyDrive/spectral_comparison_6panel.png', dpi=300, bbox_inches='tight')
print("Visualization saved: spectral_comparison_6panel.png")
```

## Implementation Instructions for External Agent

### Step-by-Step Process:

1. **Backup Original**: Create backup of original notebook before any modifications

2. **Identify Training Cell**: 
   - Search for the cell containing main training loop
   - This will likely be a code cell around position 15-20
   - Look for patterns like `for epoch in range(num_epochs):`

3. **Modify Training Cell**:
   - Use NotebookEdit tool to REPLACE the existing training cell content
   - **DO NOT ADD** a new cell - modify the existing one
   - Use the complete 6-model training code provided above

4. **Add Metrics Analysis Cell**:
   - Use NotebookEdit with `edit_mode=insert` to add new cell
   - Insert after the modified training cell
   - Use `cell_type=code`
   - Insert complete metrics analysis code

5. **Add Visualization Cell**:
   - Use NotebookEdit with `edit_mode=insert` to add new cell  
   - Insert after the metrics analysis cell
   - Use `cell_type=code`
   - Insert complete visualization code

### Critical Success Factors:

- **Test each modification** before proceeding
- **Verify cell count** remains reasonable (should be ~32 total)
- **Check for syntax errors** in inserted code
- **Ensure imports work** in Colab environment
- **Validate file paths** for saving results

### Error Prevention:

- Never use `python -c` commands
- Always use proper JSON structure for notebook cells
- Don't create duplicate training sections
- Preserve existing cell numbering where possible
- Test that all tensor operations use correct device

### Expected Outcome:

- 6 trained models stored in `trained_models` dictionary
- Comprehensive metrics comparison table with 20+ metrics
- Clean 6-panel visualization showing all concentration holdouts
- All results saved to Google Drive for download

This modification will transform the notebook from a single-model demonstration into a comprehensive validation framework for the Geodesic-Coupled Spectral NODE system.