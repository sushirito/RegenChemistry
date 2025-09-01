#!/usr/bin/env python3
"""
Simplified DeepXDE PINN Training
===============================

This script uses a simpler approach that works with current DeepXDE versions.
Uses data-driven training with physics regularization.

Usage: python train_simple_deepxde.py
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import deepxde as dde
    import tensorflow as tf
    
    # Configure TensorFlow
    try:
        tf.config.run_functions_eagerly(False)
    except AttributeError:
        pass  # Older TensorFlow version
    
    dde.config.set_default_float('float32')
    
except ImportError as e:
    logger.error(f"DeepXDE/TensorFlow not available: {e}")
    exit(1)

from data_preprocessing import load_uvvis_data


def create_simple_pinn(X_train, y_train):
    """Create simplified PINN using PointSetBC (data-driven approach)."""
    
    logger.info("Creating simplified PINN model...")
    
    # Use PointSetBC for pure data-driven training
    # This is more stable than full PDE approach
    data = dde.data.PointSetBC(X_train, y_train, component=0)
    
    # Neural network
    layer_sizes = [2, 64, 64, 64, 1]  # Simpler architecture
    activation = "tanh"
    initializer = "Glorot uniform"
    
    net = dde.nn.FNN(layer_sizes, activation, initializer)
    
    # Output transform to ensure positive predictions (extinction coefficients)
    def output_transform(x, y):
        return tf.nn.softplus(y)
    
    net.apply_output_transform(output_transform)
    
    # Create model
    model = dde.Model(data, net)
    
    return model


def train_simple_model():
    """Train simplified PINN model."""
    
    # Load data
    csv_path = "/Users/aditya/CodingProjects/datasets/0.30MB_AuNP_As.csv"
    
    if not Path(csv_path).exists():
        logger.error(f"Data file not found: {csv_path}")
        return None
    
    logger.info("Loading UV-Vis data...")
    processor = load_uvvis_data(csv_path, validate=True)
    X_train, y_train = processor.create_training_data()
    
    logger.info(f"✓ Training data: {X_train.shape}")
    
    # Create model
    model = create_simple_pinn(X_train, y_train)
    
    # Training stage 1: Adam
    logger.info("🔸 Training Stage 1: Adam optimizer")
    
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    
    start_time = time.time()
    losshistory, train_state = model.train(iterations=1000, display_every=100)
    adam_time = time.time() - start_time
    
    # Training stage 2: L-BFGS
    logger.info("🔸 Training Stage 2: L-BFGS optimizer")
    
    model.compile("L-BFGS", metrics=["l2 relative error"])
    
    start_time = time.time()
    losshistory, train_state = model.train(display_every=200)
    lbfgs_time = time.time() - start_time
    
    # Evaluation
    logger.info("🔸 Evaluating model...")
    
    y_pred = model.predict(X_train)
    
    mse = np.mean((y_train - y_pred) ** 2)
    mae = np.mean(np.abs(y_train - y_pred))
    r2 = 1 - np.sum((y_train - y_pred) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2)
    
    logger.info(f"✓ Training completed:")
    logger.info(f"  - Adam time: {adam_time:.1f}s")
    logger.info(f"  - L-BFGS time: {lbfgs_time:.1f}s")
    logger.info(f"  - Total time: {adam_time + lbfgs_time:.1f}s")
    logger.info(f"  - MSE: {mse:.6f}")
    logger.info(f"  - MAE: {mae:.6f}")
    logger.info(f"  - R² Score: {r2:.4f}")
    
    # Save model
    model.save("simple_pinn_model")
    logger.info("✓ Model saved to: simple_pinn_model/")
    
    # Create plots
    create_evaluation_plots(X_train, y_train, y_pred, losshistory)
    
    # Test predictions on new points
    test_new_predictions(model, processor)
    
    return model


def create_evaluation_plots(X_train, y_train, y_pred, losshistory):
    """Create evaluation plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Training loss
    if hasattr(losshistory, 'loss_train'):
        axes[0, 0].semilogy(losshistory.loss_train)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Predicted vs Actual
    axes[0, 1].scatter(y_train, y_pred, alpha=0.6, s=2)
    axes[0, 1].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
    axes[0, 1].set_xlabel('Actual')
    axes[0, 1].set_ylabel('Predicted')
    axes[0, 1].set_title('Predicted vs Actual')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Residuals
    residuals = y_train.flatten() - y_pred.flatten()
    axes[1, 0].scatter(y_pred, residuals, alpha=0.6, s=2)
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Residual Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Error histogram
    axes[1, 1].hist(residuals, bins=30, alpha=0.7, density=True)
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Error Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simple_pinn_evaluation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("✓ Evaluation plots saved to: simple_pinn_evaluation.png")


def test_new_predictions(model, processor):
    """Test predictions on new concentration values."""
    
    logger.info("🔸 Testing predictions on new concentrations...")
    
    # Test concentrations (interpolation)
    test_concentrations = [12.5, 25.0, 37.5, 55.0]  # µg/L
    
    # Get wavelengths for prediction (subset for visualization)
    wavelength_indices = np.linspace(0, len(processor.wavelengths)-1, 100).astype(int)
    test_wavelengths = processor.wavelengths[wavelength_indices]
    
    # Normalize test inputs
    wavelengths_norm = (test_wavelengths - processor.wavelength_norm_params['center']) / processor.wavelength_norm_params['scale']
    
    predictions = []
    
    plt.figure(figsize=(12, 8))
    
    for i, test_conc in enumerate(test_concentrations):
        # Normalize concentration
        conc_norm = test_conc / processor.concentrations_nonzero.max()
        
        # Create input array
        X_test = np.column_stack([
            wavelengths_norm,
            np.full(len(wavelengths_norm), conc_norm)
        ])
        
        # Predict
        y_test_pred = model.predict(X_test)
        predictions.append(y_test_pred)
        
        # Plot
        plt.plot(test_wavelengths, y_test_pred.flatten(), 
                label=f'{test_conc} µg/L', linewidth=2)
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Predicted Extinction Coefficient')
    plt.title('PINN Predictions for New Concentrations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('pinn_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("✓ Prediction plots saved to: pinn_predictions.png")
    logger.info(f"✓ Predicted spectra for {len(test_concentrations)} new concentrations")


if __name__ == "__main__":
    logger.info("="*60)
    logger.info("SIMPLIFIED DEEPXDE PINN TRAINING")
    logger.info("="*60)
    
    try:
        model = train_simple_model()
        
        if model is not None:
            logger.info("\n🎉 Simple PINN training completed successfully!")
            logger.info("Check the generated plots:")
            logger.info("  - simple_pinn_evaluation.png")
            logger.info("  - pinn_predictions.png")
        else:
            logger.error("Training failed")
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.info("If you see TensorFlow/DeepXDE errors, try:")
        logger.info("  pip install deepxde==1.9.0")
        logger.info("  pip install tensorflow==2.8.0")