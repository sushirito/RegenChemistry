"""
Training visualization and plotting utilities
Creates training curves and convergence plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import os


def plot_training_history(
    csv_path: str,
    save_path: Optional[str] = None,
    title_prefix: str = "Training",
    show_plot: bool = True
) -> plt.Figure:
    """
    Plot training history from CSV file
    
    Args:
        csv_path: Path to training history CSV
        save_path: Optional path to save figure
        title_prefix: Prefix for plot title
        show_plot: Whether to display the plot
        
    Returns:
        Matplotlib figure
    """
    # Load history
    df = pd.read_csv(csv_path)
    
    # Extract model info if available
    model_idx = df['model_idx'].iloc[0] if 'model_idx' in df.columns else None
    excluded_conc = df['excluded_concentration'].iloc[0] if 'excluded_concentration' in df.columns else None
    
    epochs = df['epoch'].values if 'epoch' in df.columns else np.arange(len(df))
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes.ravel()
    
    # 1. Total Loss
    ax[0].plot(epochs, df['total_loss'].values, linewidth=2, color='blue')
    ax[0].set_title("Total Loss", fontsize=12)
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].grid(True, alpha=0.3)
    
    # 2. Loss Components (if available)
    loss_components = []
    component_names = []
    if 'reconstruction_loss' in df.columns:
        loss_components.append(df['reconstruction_loss'].values)
        component_names.append('Reconstruction')
    if 'smoothness_loss' in df.columns:
        loss_components.append(df['smoothness_loss'].values)
        component_names.append('Smoothness')
    if 'path_length_loss' in df.columns:
        loss_components.append(df['path_length_loss'].values)
        component_names.append('Path Length')
    if 'bounds_loss' in df.columns:
        loss_components.append(df['bounds_loss'].values)
        component_names.append('Bounds')
    
    if loss_components:
        for comp, name in zip(loss_components, component_names):
            if not pd.isna(comp).all():
                ax[1].plot(epochs, comp, label=name, linewidth=1.5)
        ax[1].set_title("Loss Components", fontsize=12)
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Loss")
        ax[1].legend(loc='best')
        ax[1].grid(True, alpha=0.3)
    
    # 3. Learning Rates
    if 'lr_metric' in df.columns and 'lr_flow' in df.columns:
        ax[2].plot(epochs, df['lr_metric'].values, label='Metric Network', linewidth=1.5)
        ax[2].plot(epochs, df['lr_flow'].values, label='Flow Network', linewidth=1.5)
        ax[2].set_yscale('log')
        ax[2].set_title("Learning Rates", fontsize=12)
        ax[2].set_xlabel("Epoch")
        ax[2].set_ylabel("Learning Rate")
        ax[2].legend(loc='best')
        ax[2].grid(True, alpha=0.3)
    
    # 4. Convergence Rate
    if 'convergence_rate' in df.columns:
        ax[3].plot(epochs, df['convergence_rate'].values, linewidth=1.5, color='green')
        ax[3].set_ylim(0, 1)
        ax[3].set_title("Convergence Rate", fontsize=12)
        ax[3].set_xlabel("Epoch")
        ax[3].set_ylabel("Rate")
        ax[3].grid(True, alpha=0.3)
        
        # Add epoch time as secondary axis if available
        if 'epoch_time' in df.columns and not pd.isna(df['epoch_time']).all():
            ax2 = ax[3].twinx()
            ax2.plot(epochs, df['epoch_time'].values, linestyle='--', 
                    linewidth=1.5, color='orange', alpha=0.7)
            ax2.set_ylabel("Time per epoch (s)", color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
    
    # Overall title
    if model_idx is not None:
        concentration_values = [0, 10, 20, 30, 40, 60]
        excluded_ppb = concentration_values[model_idx]
        fig.suptitle(f"{title_prefix} - Model {model_idx} (Excluded: {excluded_ppb} ppb)", 
                    fontsize=14, fontweight='bold')
    else:
        fig.suptitle(f"{title_prefix} History", fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Training plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig


def plot_all_training_histories(
    log_dir: str = 'outputs/training_logs',
    save_dir: str = 'outputs/visualizations',
    show_plots: bool = False
) -> List[plt.Figure]:
    """
    Plot all training histories from a directory
    
    Args:
        log_dir: Directory containing training history CSV files
        save_dir: Directory to save plots
        show_plots: Whether to display each plot
        
    Returns:
        List of matplotlib figures
    """
    import glob
    
    # Find all training history CSVs
    csv_files = glob.glob(os.path.join(log_dir, 'training_history_model_*.csv'))
    csv_files.sort()
    
    figures = []
    
    print(f"\n📊 Creating training plots for {len(csv_files)} models...")
    
    for csv_path in csv_files:
        # Extract model index from filename
        basename = os.path.basename(csv_path)
        model_idx = basename.split('_')[-1].replace('.csv', '')
        
        # Create plot
        save_path = os.path.join(save_dir, f'training_curves_model_{model_idx}.png')
        
        fig = plot_training_history(
            csv_path=csv_path,
            save_path=save_path,
            title_prefix=f"Training Model {model_idx}",
            show_plot=show_plots
        )
        
        figures.append(fig)
    
    print(f"✅ Created {len(figures)} training plots")
    
    return figures


def create_combined_loss_plot(
    log_dir: str = 'outputs/training_logs',
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Create a combined plot showing all models' losses
    
    Args:
        log_dir: Directory containing training history CSVs
        save_path: Optional path to save figure
        show_plot: Whether to display the plot
        
    Returns:
        Matplotlib figure
    """
    import glob
    
    # Find all training history CSVs
    csv_files = glob.glob(os.path.join(log_dir, 'training_history_model_*.csv'))
    csv_files.sort()
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    concentration_values = [0, 10, 20, 30, 40, 60]
    colors = plt.cm.viridis(np.linspace(0, 1, len(csv_files)))
    
    for i, csv_path in enumerate(csv_files):
        df = pd.read_csv(csv_path)
        epochs = df['epoch'].values if 'epoch' in df.columns else np.arange(len(df))
        
        # Extract model index
        basename = os.path.basename(csv_path)
        model_idx = int(basename.split('_')[-1].replace('.csv', ''))
        excluded_ppb = concentration_values[model_idx] if model_idx < len(concentration_values) else model_idx
        
        # Plot total loss
        axes[0].plot(epochs, df['total_loss'].values, 
                    label=f'Exclude {excluded_ppb} ppb',
                    linewidth=1.5, color=colors[i])
        
        # Plot convergence rate if available
        if 'convergence_rate' in df.columns:
            axes[1].plot(epochs, df['convergence_rate'].values,
                        label=f'Exclude {excluded_ppb} ppb',
                        linewidth=1.5, color=colors[i])
    
    # Configure subplots
    axes[0].set_title("Total Loss - All Models", fontsize=12)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title("Convergence Rate - All Models", fontsize=12)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Convergence Rate")
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc='best', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle("Training Progress - All Leave-One-Out Models", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Combined plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig


def create_metric_comparison_bar_plot(
    metrics_csv: str,
    top_k: int = 10,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Create bar plot comparing top metrics between methods
    
    Args:
        metrics_csv: Path to metrics CSV
        top_k: Number of top metrics to show
        save_path: Optional path to save figure
        show_plot: Whether to display the plot
        
    Returns:
        Matplotlib figure
    """
    # Load metrics
    df = pd.read_csv(metrics_csv)
    
    # Separate by method
    geo_df = df[df['Method'] == 'Geodesic']
    basic_df = df[df['Method'] == 'Basic']
    
    # Get metric columns (exclude metadata)
    metric_cols = [col for col in df.columns 
                  if col not in ['Concentration_ppb', 'Method']]
    
    # Calculate mean improvements
    improvements = {}
    for metric in metric_cols:
        geo_mean = geo_df[metric].mean()
        basic_mean = basic_df[metric].mean()
        
        # Calculate improvement based on metric type
        from geodesic_m1.utils.metrics import HIGHER_BETTER, LOWER_BETTER
        
        if metric in HIGHER_BETTER:
            improvements[metric] = geo_mean - basic_mean
        elif metric in LOWER_BETTER:
            improvements[metric] = basic_mean - geo_mean
        else:
            improvements[metric] = 0
    
    # Sort and get top k
    sorted_metrics = sorted(improvements.items(), key=lambda x: abs(x[1]), reverse=True)
    top_metrics = sorted_metrics[:top_k]
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = [m[0] for m in top_metrics]
    values = [m[1] for m in top_metrics]
    colors = ['green' if v > 0 else 'red' for v in values]
    
    bars = ax.bar(range(len(metrics)), values, color=colors, alpha=0.7, edgecolor='black')
    
    # Customize plot
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.set_xlabel("Metric")
    ax.set_ylabel("Improvement (Geodesic vs Basic)")
    ax.set_title(f"Top {top_k} Metric Improvements", fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., 
               height + (0.01 * max(abs(min(values)), max(values)) if height > 0 else -0.01 * max(abs(min(values)), max(values))),
               f'{val:.4f}',
               ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Metric comparison saved to {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig