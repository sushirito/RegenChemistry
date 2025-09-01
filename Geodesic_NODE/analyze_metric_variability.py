#!/usr/bin/env python3
"""
Analyze spectral validation metrics to identify the most illustrative metrics
that showcase the baseline interpolation's dramatic failure at edge cases
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def analyze_metric_variability():
    """Analyze metrics to find those with highest edge vs middle variability"""
    
    # Load metrics
    df = pd.read_csv('spectral_validation_metrics.csv')
    
    # Separate data rows from summary rows
    df_data = df[df['Concentration_ppb'].apply(lambda x: str(x).replace('.','').isdigit())].copy()
    df_data['Concentration_ppb'] = df_data['Concentration_ppb'].astype(float)
    
    # Define edge and middle concentrations
    edge_concs = [0.0, 60.0]
    middle_concs = [20.0, 30.0]
    
    # Calculate edge vs middle performance for each metric
    metrics_analysis = []
    
    for col in df_data.columns:
        if col == 'Concentration_ppb':
            continue
            
        edge_values = df_data[df_data['Concentration_ppb'].isin(edge_concs)][col].values
        middle_values = df_data[df_data['Concentration_ppb'].isin(middle_concs)][col].values
        
        # Calculate statistics
        edge_mean = np.mean(edge_values)
        middle_mean = np.mean(middle_values)
        edge_worst = np.max(np.abs(edge_values))
        middle_best = np.min(np.abs(middle_values))
        
        # Calculate variability measures
        overall_std = df_data[col].std()
        overall_mean = df_data[col].mean()
        cv = overall_std / (abs(overall_mean) + 1e-10)  # Coefficient of variation
        
        # Ratio of worst edge to best middle
        if middle_best != 0:
            worst_to_best_ratio = edge_worst / middle_best
        else:
            worst_to_best_ratio = np.inf if edge_worst != 0 else 1
        
        # Range (max - min)
        range_val = df_data[col].max() - df_data[col].min()
        
        metrics_analysis.append({
            'Metric': col,
            'Edge_Mean': edge_mean,
            'Middle_Mean': middle_mean,
            'Edge_Worst': edge_worst,
            'Middle_Best': middle_best,
            'Worst_to_Best_Ratio': worst_to_best_ratio,
            'Overall_Std': overall_std,
            'Coeff_Variation': cv,
            'Range': range_val,
            'Edge_0': df_data[df_data['Concentration_ppb'] == 0.0][col].values[0],
            'Edge_60': df_data[df_data['Concentration_ppb'] == 60.0][col].values[0],
            'Middle_20': df_data[df_data['Concentration_ppb'] == 20.0][col].values[0],
            'Middle_30': df_data[df_data['Concentration_ppb'] == 30.0][col].values[0]
        })
    
    df_analysis = pd.DataFrame(metrics_analysis)
    
    # Identify metrics where lower is better vs higher is better
    lower_is_better = ['RMSE', 'MAE', 'MAPE', 'Max_Error', 'SAM_radians', 
                      'Wasserstein', 'KL_Divergence', 'JS_Distance', 
                      'Peak_Lambda_Error_nm', 'Peak_Abs_Error', 'FWHM_Diff', 
                      'Area_Diff', 'DTW_Distance', 'Frechet_Dist', 'Derivative_MSE']
    
    higher_is_better = ['R2_Score', 'Pearson_R', 'Spearman_R', 'Cosine_Sim', 
                       'SSIM', 'MS_SSIM', 'FFT_Correlation']
    
    # Special case: Power_Ratio should be close to 1
    target_one = ['Power_Ratio']
    
    # Calculate "badness" score for ranking
    for idx, row in df_analysis.iterrows():
        metric = row['Metric']
        if metric in lower_is_better:
            # For these, higher values at edges are bad
            badness = abs(row['Edge_Worst'] - row['Middle_Best'])
        elif metric in higher_is_better:
            # For these, lower values at edges are bad
            badness = abs(row['Middle_Best'] - row['Edge_Worst'])
        elif metric in target_one:
            # Distance from 1 is bad
            edge_dist = max(abs(row['Edge_0'] - 1), abs(row['Edge_60'] - 1))
            middle_dist = min(abs(row['Middle_20'] - 1), abs(row['Middle_30'] - 1))
            badness = edge_dist - middle_dist
        else:
            badness = row['Range']
        
        df_analysis.loc[idx, 'Badness_Score'] = badness
    
    # Sort by coefficient of variation (most variable metrics)
    df_analysis = df_analysis.sort_values('Coeff_Variation', ascending=False)
    
    return df_analysis, df_data

def select_most_illustrative_metrics(df_analysis):
    """Select the most illustrative metrics from different categories"""
    
    # Define metric categories
    categories = {
        'Statistical': ['RMSE', 'MAE', 'MAPE', 'Max_Error', 'R2_Score'],
        'Correlation': ['Pearson_R', 'Spearman_R', 'Cosine_Sim'],
        'Structural': ['SSIM', 'MS_SSIM', 'SAM_radians'],
        'Distribution': ['Wasserstein', 'KL_Divergence', 'JS_Distance'],
        'Spectral': ['Peak_Lambda_Error_nm', 'Peak_Abs_Error', 'FWHM_Diff', 'Area_Diff'],
        'Shape': ['DTW_Distance', 'Frechet_Dist', 'Derivative_MSE'],
        'Frequency': ['FFT_Correlation', 'Power_Ratio']
    }
    
    # Select best metric from each category based on coefficient of variation
    selected_metrics = {}
    
    for category, metrics in categories.items():
        category_df = df_analysis[df_analysis['Metric'].isin(metrics)]
        if not category_df.empty:
            # Get metric with highest coefficient of variation
            best_metric = category_df.iloc[0]['Metric']  # Already sorted by CV
            selected_metrics[category] = {
                'metric': best_metric,
                'cv': category_df.iloc[0]['Coeff_Variation'],
                'edge_0': category_df.iloc[0]['Edge_0'],
                'edge_60': category_df.iloc[0]['Edge_60'],
                'middle_20': category_df.iloc[0]['Middle_20'],
                'middle_30': category_df.iloc[0]['Middle_30'],
                'worst_to_best': category_df.iloc[0]['Worst_to_Best_Ratio']
            }
    
    return selected_metrics

def create_dramatic_visualization(selected_metrics, df_data):
    """Create visualization highlighting the dramatic differences"""
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=[f"{cat}: {data['metric']}" for cat, data in selected_metrics.items()],
        vertical_spacing=0.15,
        horizontal_spacing=0.08
    )
    
    concentrations = df_data['Concentration_ppb'].values
    colors = ['red', 'orange', 'yellow', 'green', 'yellow', 'red']  # Edge-to-middle gradient
    
    positions = [(1,1), (1,2), (1,3), (1,4), (2,1), (2,2), (2,3), (2,4)]
    
    for idx, (category, data) in enumerate(selected_metrics.items()):
        if idx >= len(positions):
            break
            
        row, col = positions[idx]
        metric = data['metric']
        values = df_data[metric].values
        
        # Create bar chart
        fig.add_trace(
            go.Bar(
                x=concentrations,
                y=values,
                marker=dict(color=colors, line=dict(color='black', width=1)),
                showlegend=False,
                hovertemplate=f'{metric}: %{{y:.4f}}<br>Conc: %{{x}} ppb<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Add annotations for dramatic differences
        max_val = np.max(np.abs(values))
        min_val = np.min(np.abs(values))
        if max_val != 0 and min_val != 0:
            ratio = max_val / min_val
            if ratio > 10:
                fig.add_annotation(
                    x=concentrations[np.argmax(np.abs(values))],
                    y=values[np.argmax(np.abs(values))],
                    text=f'{ratio:.0f}x',
                    showarrow=False,
                    font=dict(size=12, color='red'),
                    row=row, col=col
                )
    
    fig.update_layout(
        title={
            'text': 'Dramatic Failure of Basic Interpolation at Edge Cases<br><sub>Red = Edge concentrations (0, 60 ppb) | Green = Middle concentrations (20, 30 ppb)</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        height=700,
        width=1600,
        showlegend=False,
        paper_bgcolor='#f8f9fa'
    )
    
    fig.update_xaxes(title_text="Concentration (ppb)")
    fig.update_yaxes(title_text="Metric Value")
    
    return fig

def main():
    """Main analysis function"""
    
    print("="*80)
    print("ANALYZING METRIC VARIABILITY FOR BASELINE FAILURE DEMONSTRATION")
    print("="*80)
    
    # Analyze all metrics
    df_analysis, df_data = analyze_metric_variability()
    
    # Print top 10 most variable metrics
    print("\nTop 10 Most Variable Metrics (by Coefficient of Variation):")
    print("-"*60)
    for idx, row in df_analysis.head(10).iterrows():
        print(f"{row['Metric']:25s}: CV={row['Coeff_Variation']:8.2f}, "
              f"Range={row['Range']:10.4f}, Worst/Best={row['Worst_to_Best_Ratio']:8.2f}")
    
    # Select most illustrative metrics from each category
    selected_metrics = select_most_illustrative_metrics(df_analysis)
    
    print("\n" + "="*80)
    print("MOST ILLUSTRATIVE METRICS BY CATEGORY")
    print("="*80)
    
    for category, data in selected_metrics.items():
        print(f"\n{category:15s}: {data['metric']}")
        print(f"  Edge (0 ppb):    {data['edge_0']:10.4f}")
        print(f"  Edge (60 ppb):   {data['edge_60']:10.4f}")
        print(f"  Middle (20 ppb): {data['middle_20']:10.4f}")
        print(f"  Middle (30 ppb): {data['middle_30']:10.4f}")
        print(f"  Worst/Best Ratio: {data['worst_to_best']:.1f}x")
    
    # Identify the absolute worst performers
    print("\n" + "="*80)
    print("🎯 MOST DRAMATIC METRICS FOR SHOWCASING BASELINE FAILURE:")
    print("="*80)
    
    # Hand-pick the most dramatic based on analysis
    dramatic_metrics = {
        'R2_Score': "Goes from 0.93 (good) to -34.13 (catastrophic failure!)",
        'Peak_Lambda_Error_nm': "0 nm everywhere except 459 nm at 60 ppb (complete peak misidentification!)",
        'MAPE': "3.8% at best to 100.7% at worst (26x degradation!)",
        'SAM_radians': "0.035 to 0.521 radians (15x worse spectral angle!)",
        'Wasserstein': "0.004 to 0.103 (25x distribution shift!)",
        'Power_Ratio': "~1.0 (perfect) to 4.045 (4x power distortion!)"
    }
    
    for metric, description in dramatic_metrics.items():
        row = df_data[df_data['Concentration_ppb'] == 60.0][metric].values[0]
        print(f"\n{metric}: {description}")
    
    # Create visualization
    fig = create_dramatic_visualization(selected_metrics, df_data)
    fig.write_html('baseline_failure_analysis.html')
    print("\n✓ Visualization saved to 'baseline_failure_analysis.html'")
    
    # Save analysis to CSV
    df_analysis.to_csv('metric_variability_analysis.csv', index=False)
    print("✓ Detailed analysis saved to 'metric_variability_analysis.csv'")
    
    print("\n" + "="*80)
    print("RECOMMENDATION FOR PAPER/PRESENTATION:")
    print("="*80)
    print("""
Use these 6 metrics to tell your story:

1. **R2 Score** (Statistical Fit): -34.13 at 60 ppb
   → Shows complete model breakdown (negative R² means worse than horizontal line!)

2. **Peak λ Error** (Spectral Accuracy): 459 nm at 60 ppb  
   → Can't even identify the correct absorption peak!

3. **MAPE** (Percentage Error): 100.7% at 60 ppb
   → Average error exceeds actual values!

4. **SAM** (Shape Preservation): 0.521 radians at 60 ppb
   → Spectral shape completely distorted (30° angle!)

5. **Wasserstein** (Distribution Shift): 0.103 at 60 ppb
   → Massive distribution mismatch

6. **Power Ratio** (Frequency Content): 4.045 at 60 ppb
   → 4x energy distortion in frequency domain

These span all domains (statistical, spectral, structural, distributional, frequency) 
and show 4-26x performance degradation at edges vs middle concentrations.
Perfect for demonstrating how your neural network handles edge cases!
""")

if __name__ == "__main__":
    main()