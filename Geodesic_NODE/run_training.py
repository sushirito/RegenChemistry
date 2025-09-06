#!/usr/bin/env python3
"""
Main Entry Point for Training MPS-Optimized Geodesic NODE
"""

import argparse
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from geodesic_mps.configs.train_config import (
    ExperimentConfig,
    get_quick_test_config,
    get_full_training_config,
    get_benchmark_config
)
from geodesic_mps.training.train import Trainer


def main():
    """Main training entry point"""
    parser = argparse.ArgumentParser(description="Train Geodesic Spectral NODE")
    parser.add_argument(
        '--mode', 
        type=str, 
        default='quick',
        choices=['quick', 'full', 'benchmark', 'custom'],
        help='Training mode'
    )
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--device', type=str, help='Override device (mps/cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations after training')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Get configuration
    if args.mode == 'quick':
        config = get_quick_test_config()
        print("Using quick test configuration")
    elif args.mode == 'full':
        config = get_full_training_config()
        print("Using full training configuration")
    elif args.mode == 'benchmark':
        config = get_benchmark_config()
        print("Using benchmark configuration")
    else:
        config = ExperimentConfig()
        print("Using custom configuration")
    
    # Override parameters if specified
    if args.epochs is not None:
        config.training.n_epochs = args.epochs
    if args.batch_size is not None:
        config.training.micro_batch_size = args.batch_size
    if args.lr is not None:
        config.training.metric_lr = args.lr
        config.training.spectral_lr = args.lr * 2
    if args.device is not None:
        config.mps.device = args.device
    
    # Print configuration
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION")
    print("="*50)
    config_dict = config.to_dict()
    for key, value in config_dict.items():
        if isinstance(value, dict):
            print(f"\n{key.upper()}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    print("="*50 + "\n")
    
    # Create trainer and train
    trainer = Trainer(config)
    
    try:
        trainer.train()
        print("\n✅ Training completed successfully!")
        
        # Generate visualizations if requested
        if args.visualize:
            print("\n📊 Generating visualizations...")
            from geodesic_mps.utils.visualization import visualize_training_results
            visualize_training_results()
            
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        trainer.save_checkpoint()
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()