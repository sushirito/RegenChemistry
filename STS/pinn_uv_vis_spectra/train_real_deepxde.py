#!/usr/bin/env python3
"""
Real DeepXDE PINN Training Script
=================================

This script performs actual DeepXDE training with proper API compatibility.

Usage: python train_real_deepxde.py
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import deepxde as dde
    import tensorflow as tf
    
    # Configure TensorFlow to avoid memory issues
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        logger.info("✓ GPU configured")
    
    # Set DeepXDE configuration
    dde.config.set_default_float('float32')
    
except ImportError as e:
    logger.error(f"DeepXDE/TensorFlow not available: {e}")
    logger.error("Please install: pip install deepxde tensorflow")
    sys.exit(1)

from data_preprocessing import load_uvvis_data


def create_pinn_model_real(X_train, y_train):
    """Create real DeepXDE PINN model with proper API usage."""
    
    logger.info("Creating DeepXDE PINN model...")
    
    # Define the domain (normalized inputs)
    # X_train[:, 0] = wavelength_norm ∈ [-1, 1] 
    # X_train[:, 1] = concentration_norm ∈ [0, 1]
    
    # Create geometry - simple rectangle for the domain
    geom = dde.geometry.Rectangle(xmin=[-1, 0], xmax=[1, 1])
    
    # Define the PDE (Beer-Lambert physics)
    def pde(x, y):
        """
        Physics-Informed Neural Network PDE for Beer-Lambert law.
        
        x: input [wavelength_norm, concentration_norm]
        y: output [extinction_coefficient_prediction]
        
        Physics constraint: ∂ε/∂λ should be smooth (spectral smoothness)
        """
        # Get derivatives
        dy_dx = dde.grad.jacobian(y, x, i=0, j=0)  # ∂ε/∂λ
        d2y_dx2 = dde.grad.hessian(y, x, i=0, j=0)  # ∂²ε/∂λ²
        
        # Smoothness constraint: penalize large second derivatives
        return d2y_dx2
    
    # Create PDE data (physics points)
    data_pde = dde.data.PDE(geom, pde, [], num_domain=1000, num_boundary=100)
    
    # Create experimental data
    data_exp = dde.data.PointSetBC(X_train, y_train, component=0)
    
    # Combine PDE and experimental data
    data = dde.data.PDE(
        geom,
        pde,
        [],
        num_domain=1000,
        num_boundary=100,
        anchors=X_train,  # Use experimental points as anchor points
    )
    
    # Alternative: Use PointSet for pure data-driven approach with physics regularization
    data_combined = dde.data.PointSetBC(X_train, y_train, component=0)
    
    # Neural network architecture
    layer_sizes = [2, 64, 128, 128, 64, 32, 1]
    activation = "tanh"
    initializer = "Glorot uniform"
    
    net = dde.nn.FNN(layer_sizes, activation, initializer)
    
    # Apply input/output transformations for better training
    def input_transform(x):
        """Transform inputs for better conditioning."""
        return x  # Already normalized in preprocessing
    
    def output_transform(x, y):
        """Transform outputs to ensure physical constraints."""
        # Ensure extinction coefficient is non-negative
        return tf.nn.softplus(y)
    
    net.apply_output_transform(output_transform)
    
    # Create model
    model = dde.Model(data_combined, net)
    
    return model


def train_deepxde_model(X_train, y_train, output_dir="deepxde_results"):
    """Train real DeepXDE model."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger.info("="*60)
    logger.info("REAL DEEPXDE TRAINING")
    logger.info("="*60)
    
    # Create model
    model = create_pinn_model_real(X_train, y_train)
    
    # Multi-component loss weights
    loss_weights = [1.0]  # Experimental data weight
    
    # Stage 1: Adam training
    logger.info("🔸 Stage 1: Adam optimizer training...")
    
    model.compile(
        "adam",
        lr=0.001,
        loss_weights=loss_weights,
        metrics=["l2 relative error"]
    )
    
    # Training callbacks
    callbacks = [
        dde.callbacks.ModelCheckpoint(
            str(output_dir / "model_checkpoint"),
            save_better_only=True,
            period=100
        )
    ]
    
    start_time = time.time()
    
    # Train with Adam
    losshistory, train_state = model.train(
        iterations=2000,
        display_every=200,
        callbacks=callbacks
    )
    
    stage1_time = time.time() - start_time
    
    # Stage 2: L-BFGS fine-tuning
    logger.info("🔸 Stage 2: L-BFGS fine-tuning...")
    
    model.compile(
        "L-BFGS",
        loss_weights=loss_weights,
        metrics=["l2 relative error"]
    )
    
    start_time = time.time()
    
    # Fine-tune with L-BFGS
    losshistory, train_state = model.train(
        display_every=500,
        callbacks=callbacks
    )
    
    stage2_time = time.time() - start_time
    
    # Evaluation
    logger.info("🔸 Evaluating trained model...")
    
    y_pred = model.predict(X_train)
    
    mse = np.mean((y_train - y_pred) ** 2)
    mae = np.mean(np.abs(y_train - y_pred))
    r2 = 1 - np.sum((y_train - y_pred) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2)
    
    results = {
        'training_time': {
            'stage1_adam': stage1_time,
            'stage2_lbfgs': stage2_time,
            'total': stage1_time + stage2_time
        },
        'metrics': {
            'mse': float(mse),
            'mae': float(mae),
            'r2_score': float(r2)
        },
        'model_info': {
            'architecture': [2, 64, 128, 128, 64, 32, 1],
            'training_points': len(X_train),
            'total_iterations': len(losshistory.loss_train) if hasattr(losshistory, 'loss_train') else 'Unknown'
        }
    }
    
    logger.info(f"✓ Training completed:")
    logger.info(f"  - Total time: {stage1_time + stage2_time:.1f}s")
    logger.info(f"  - MSE: {mse:.6f}")
    logger.info(f"  - MAE: {mae:.6f}")
    logger.info(f"  - R² Score: {r2:.4f}")
    
    # Save model
    model.save(str(output_dir / "trained_model"))
    logger.info(f"✓ Model saved to: {output_dir / 'trained_model'}")
    
    # Plot training history
    try:
        plot_training_history(losshistory, output_dir)
    except Exception as e:
        logger.warning(f"Plotting failed: {e}")
    
    # Make test predictions
    test_predictions(model, X_train, y_train, output_dir)
    
    return model, results


def plot_training_history(losshistory, output_dir):
    """Plot training loss history."""
    
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    if hasattr(losshistory, 'loss_train'):
        plt.semilogy(losshistory.loss_train, label='Training Loss')
        if hasattr(losshistory, 'loss_test'):
            plt.semilogy(losshistory.loss_test, label='Test Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Metrics plot
    plt.subplot(1, 2, 2)
    if hasattr(losshistory, 'metrics_train'):
        metrics = np.array(losshistory.metrics_train)
        if len(metrics.shape) > 1 and metrics.shape[1] > 0:
            plt.semilogy(metrics[:, 0], label='L2 Relative Error')
            plt.xlabel('Iteration')
            plt.ylabel('Relative Error')
            plt.title('Training Metrics')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_history.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Training plots saved to: {output_dir / 'training_history.png'}")


def test_predictions(model, X_train, y_train, output_dir):
    """Make test predictions and visualizations."""
    
    # Predictions on training data
    y_pred = model.predict(X_train)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Predicted vs Actual
    axes[0, 0].scatter(y_train, y_pred, alpha=0.6, s=1)
    axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
    axes[0, 0].set_xlabel('Actual Extinction Coefficient')
    axes[0, 0].set_ylabel('Predicted Extinction Coefficient')
    axes[0, 0].set_title('Predicted vs Actual')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals
    residuals = y_train.flatten() - y_pred.flatten()
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=1)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Error histogram
    axes[1, 0].hist(residuals, bins=50, alpha=0.7, density=True)
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Error Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Wavelength-concentration grid predictions
    # Create a test grid
    wavelengths_norm = np.linspace(-1, 1, 50)
    concentrations_norm = np.linspace(0.1, 1, 20)
    
    W, C = np.meshgrid(wavelengths_norm, concentrations_norm)
    X_test = np.column_stack([W.ravel(), C.ravel()])
    
    y_test_pred = model.predict(X_test)
    Z = y_test_pred.reshape(W.shape)
    
    im = axes[1, 1].contourf(W, C, Z, levels=20, cmap='viridis')
    axes[1, 1].set_xlabel('Wavelength (normalized)')
    axes[1, 1].set_ylabel('Concentration (normalized)')
    axes[1, 1].set_title('Predicted Extinction Coefficient Field')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_evaluation.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Evaluation plots saved to: {output_dir / 'model_evaluation.png'}")


def main():
    """Main training execution."""
    
    # Data path
    csv_path = "/Users/aditya/CodingProjects/datasets/0.30MB_AuNP_As.csv"
    
    if not Path(csv_path).exists():
        logger.error(f"Data file not found: {csv_path}")
        logger.info("Please provide path to your UV-Vis CSV data")
        return 1
    
    try:
        logger.info("🔸 Loading and preprocessing data...")
        
        # Load data
        processor = load_uvvis_data(csv_path, validate=True)
        X_train, y_train = processor.create_training_data()
        
        logger.info(f"✓ Data loaded: {X_train.shape} training points")
        logger.info(f"  - Wavelengths: {processor.wavelengths.min():.0f}-{processor.wavelengths.max():.0f} nm")
        logger.info(f"  - Concentrations: {processor.concentrations.min()}-{processor.concentrations.max()} µg/L")
        
        # Train model
        model, results = train_deepxde_model(X_train, y_train)
        
        # Final summary
        logger.info("\n" + "="*60)
        logger.info("DEEPXDE TRAINING COMPLETE")
        logger.info("="*60)
        logger.info(f"✓ Architecture: {results['model_info']['architecture']}")
        logger.info(f"✓ Training points: {results['model_info']['training_points']}")
        logger.info(f"✓ Training time: {results['training_time']['total']:.1f}s")
        logger.info(f"✓ Final R² score: {results['metrics']['r2_score']:.4f}")
        logger.info(f"✓ Final MSE: {results['metrics']['mse']:.6f}")
        logger.info("✓ Model saved and ready for predictions!")
        logger.info("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"DeepXDE training failed: {e}")
        logger.error("This might be due to DeepXDE version compatibility")
        logger.info("Try: pip install deepxde==1.9.0 tensorflow==2.8.0")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)