#!/usr/bin/env python3
"""
M1 Mac optimized training script for Geodesic-Coupled Spectral NODE
Ultra-fast training with MPS acceleration

Usage:
    python main.py --config memory_optimized    # For 8-16GB M1 Macs
    python main.py --config performance         # For 16GB+ M1 Macs  
    python main.py --epochs 50                  # Custom epochs
"""

import argparse
import sys
import time
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent  # Go up one level to Geodesic_NODE
sys.path.insert(0, str(project_root))

from geodesic_m1.configs.m1_config import (
    create_m1_config, 
    get_memory_optimized_config, 
    get_performance_optimized_config
)
from geodesic_m1.training.trainer import M1Trainer
from geodesic_m1.training.data_loader import create_leave_one_out_datasets
from geodesic_m1.data.generator import create_full_dataset
from geodesic_m1.core.device_manager import M1DeviceManager


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='M1 Mac Geodesic NODE Training')
    
    # Configuration options
    parser.add_argument('--config', type=str, default='default',
                       choices=['default', 'memory_optimized', 'performance'],
                       help='Configuration preset')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    
    # Data parameters
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to spectral data CSV file')
    parser.add_argument('--use-synthetic', action='store_true',
                       help='Use synthetic data instead of real data')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Output directory for results')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Directory for model checkpoints')
    
    # Performance options
    parser.add_argument('--profile', action='store_true',
                       help='Enable performance profiling')
    parser.add_argument('--validate-only', action='store_true',
                       help='Run validation only (requires trained models)')
    
    # Verbosity
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    return parser.parse_args()


def setup_configuration(args):
    """Setup training configuration based on arguments"""
    # Choose base configuration
    if args.config == 'memory_optimized':
        config = get_memory_optimized_config()
    elif args.config == 'performance':
        config = get_performance_optimized_config()
    else:
        config = create_m1_config()
    
    # Override with command line arguments
    custom_params = {}
    if args.epochs is not None:
        custom_params['epochs'] = args.epochs
    if args.batch_size is not None:
        custom_params['recommended_batch_size'] = args.batch_size
    if args.lr is not None:
        custom_params['metric_lr'] = args.lr
        custom_params['flow_lr'] = args.lr * 2  # Flow network gets 2x learning rate
        
    # Apply custom parameters
    if custom_params:
        config = create_m1_config({**config.to_dict(), **custom_params})
    
    # Setup profiling
    if args.profile:
        config.profile_training = True
        
    return config


def create_directories(args):
    """Create necessary directories"""
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    return output_dir, checkpoint_dir


def main():
    """Main training function"""
    print("🚀 Geodesic NODE Training on M1 Mac")
    print("=" * 50)
    
    # Parse arguments
    args = parse_arguments()
    verbose = not args.quiet
    
    # Setup configuration
    config = setup_configuration(args)
    
    if verbose:
        config.print_config()
    
    # Create directories
    output_dir, checkpoint_dir = create_directories(args)
    
    # Initialize device manager
    device_manager = M1DeviceManager()
    if verbose:
        device_manager.print_system_info()
    
    # Generate or load dataset
    if verbose:
        print("\n📊 Loading dataset...")
        
    use_synthetic = args.use_synthetic or args.data_path is None
    wavelengths, absorbance_data = create_full_dataset(
        device=config.device,
        use_synthetic=use_synthetic,
        data_path=args.data_path
    )
    
    if verbose:
        print(f"✅ Dataset loaded: {absorbance_data.shape}")
        print(f"   Wavelengths: {len(wavelengths)} points ({wavelengths.min():.0f}-{wavelengths.max():.0f} nm)")
        print(f"   Concentrations: {absorbance_data.shape[0]} levels")
    
    # Create leave-one-out datasets
    if verbose:
        print("\n🔄 Creating leave-one-out datasets...")
        
    datasets = create_leave_one_out_datasets(
        wavelengths.cpu().numpy(),
        absorbance_data.cpu().numpy(),
        device=config.device
    )
    
    # Initialize trainer
    trainer = M1Trainer(
        device=config.device,
        checkpoint_dir=str(checkpoint_dir),
        verbose=verbose
    )
    
    start_time = time.time()
    
    # Import validation and visualization modules
    from validation.evaluator import run_complete_validation
    from visualization.training_plots import plot_all_training_histories, create_combined_loss_plot
    from visualization.comparison_3d import create_3d_comparison_plot
    
    if args.validate_only:
        # Run validation only
        if verbose:
            print("\n🧪 Running validation...")
            
        try:
            validation_results = trainer.validate_models(datasets, config.to_dict())
            
            if verbose:
                print("✅ Validation completed!")
                agg = validation_results['aggregate_results']
                print(f"   Mean R²: {agg['mean_r2']:.4f}")
                print(f"   Mean RMSE: {agg['mean_rmse']:.4f}")
                
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return 1
            
    else:
        # Run full training
        if verbose:
            print(f"\n🏋️ Training 6 models with leave-one-out validation...")
            print(f"   Expected time: {config.epochs * 6 * 0.5 / 60:.1f} - {config.epochs * 6 * 1.5 / 60:.1f} hours")
            
        try:
            # Train all models
            training_results = trainer.train_all_models(datasets, config.to_dict())
            
            if verbose:
                print("✅ Training completed!")
                print(f"   Total time: {training_results['total_training_time_hours']:.2f} hours")
                print(f"   Mean best loss: {training_results['mean_best_loss']:.6f}")
            
            # Run validation
            if verbose:
                print("\n🧪 Running validation...")
                
            validation_results = trainer.validate_models(datasets, config.to_dict())
            
            if verbose:
                print("✅ Validation completed!")
                agg = validation_results['aggregate_results']
                print(f"   Mean R²: {agg['mean_r2']:.4f}")
                print(f"   Mean RMSE: {agg['mean_rmse']:.4f}")
                
            # Save results
            results_file = output_dir / "training_results.pt"
            torch.save({
                'training_results': training_results,
                'validation_results': validation_results,
                'config': config.to_dict()
            }, results_file)
            
            if verbose:
                print(f"💾 Results saved to {results_file}")
            
            # Run comprehensive validation with metrics
            if verbose:
                print("\n🔬 Running comprehensive validation with 20+ metrics...")
            
            # Prepare model paths
            model_paths = {}
            for i in range(6):
                model_path = checkpoint_dir / f"best_model_{i}.pt"
                if model_path.exists():
                    model_paths[i] = str(model_path)
            
            # Run validation if we have models
            if model_paths:
                # Get data path
                data_path = args.data_path if args.data_path else "data/spectral_data.csv"
                
                # Run validation
                metrics_df, predictions_df = run_complete_validation(
                    model_paths=model_paths,
                    data_path=data_path,
                    device=config.device,
                    save_metrics=True,
                    save_predictions=True
                )
                
                # Create visualizations
                if verbose:
                    print("\n📊 Creating visualizations...")
                
                # Plot all training histories
                plot_all_training_histories(
                    log_dir=str(output_dir / "training_logs"),
                    save_dir=str(output_dir / "visualizations"),
                    show_plots=False
                )
                
                # Create combined loss plot
                combined_plot_path = output_dir / "visualizations" / "combined_training.png"
                create_combined_loss_plot(
                    log_dir=str(output_dir / "training_logs"),
                    save_path=str(combined_plot_path),
                    show_plot=False
                )
                
                # Create 3D comparison plot
                viz_3d_path = output_dir / "visualizations" / "comparison_3d.html"
                create_3d_comparison_plot(
                    predictions_csv=str(output_dir / "predictions" / "validation_predictions.csv"),
                    save_path=str(viz_3d_path),
                    show_plot=False
                )
                
                if verbose:
                    print(f"✅ Visualizations saved to {output_dir / 'visualizations'}")
                
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            if verbose:
                traceback.print_exc()
            return 1
    
    total_time = time.time() - start_time
    
    if verbose:
        print(f"\n🎉 Complete! Total time: {total_time/3600:.2f} hours")
        
        # Memory summary
        device_manager.monitor_memory_usage("Final")
        
        # Training summary if available
        if hasattr(trainer, 'training_history'):
            summary = trainer.get_training_summary()
            print(f"📈 Training Summary:")
            print(f"   Models trained: {summary['n_models']}")
            print(f"   Total parameters: {summary['total_parameters']:,}")
            print(f"   Device: {summary['device']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())