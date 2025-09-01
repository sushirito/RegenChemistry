#!/usr/bin/env python3
"""
Comprehensive Spectral Validation Metrics
Calculates 20+ metrics to quantify interpolation errors for leave-one-out validation
"""

import pandas as pd
import numpy as np
from scipy import stats, signal, spatial
from scipy.interpolate import interp1d
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import directed_hausdorff
from skimage.metrics import structural_similarity as ssim
import warnings
warnings.filterwarnings('ignore')

def load_spectral_data(filepath):
    """Load and reshape spectral data"""
    df = pd.read_csv(filepath)
    wavelengths = df['Wavelength'].values
    concentrations = [float(col) for col in df.columns[1:]]
    absorbance_matrix = df.iloc[:, 1:].values
    return wavelengths, concentrations, absorbance_matrix

def interpolate_holdout_concentration(wavelengths, concentrations, absorbance_matrix, holdout_idx):
    """Interpolate the held-out concentration using remaining data"""
    train_concs = concentrations[:holdout_idx] + concentrations[holdout_idx+1:]
    train_abs = np.column_stack([absorbance_matrix[:, i] for i in range(len(concentrations)) if i != holdout_idx])
    
    interpolated_abs = np.zeros(len(wavelengths))
    holdout_conc = concentrations[holdout_idx]
    
    for i, wl in enumerate(wavelengths):
        if len(train_concs) > 3:
            interp_func = interp1d(train_concs, train_abs[i, :], kind='cubic', 
                                  fill_value='extrapolate', bounds_error=False)
        else:
            interp_func = interp1d(train_concs, train_abs[i, :], kind='linear',
                                  fill_value='extrapolate', bounds_error=False)
        interpolated_abs[i] = interp_func(holdout_conc)
    
    return interpolated_abs

def calculate_basic_metrics(actual, predicted):
    """Calculate basic statistical metrics"""
    metrics = {}
    
    # Basic errors
    metrics['RMSE'] = np.sqrt(np.mean((actual - predicted) ** 2))
    metrics['MAE'] = np.mean(np.abs(actual - predicted))
    
    # Avoid division by zero in MAPE
    non_zero_mask = actual != 0
    if np.any(non_zero_mask):
        metrics['MAPE'] = np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100
    else:
        metrics['MAPE'] = np.nan
    
    metrics['Max_Error'] = np.max(np.abs(actual - predicted))
    
    # R-squared
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    metrics['R2_Score'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return metrics

def calculate_correlation_metrics(actual, predicted):
    """Calculate correlation and similarity metrics"""
    metrics = {}
    
    # Correlations
    metrics['Pearson_R'] = np.corrcoef(actual, predicted)[0, 1]
    metrics['Spearman_R'] = stats.spearmanr(actual, predicted)[0]
    
    # Cosine similarity
    dot_product = np.dot(actual, predicted)
    norm_actual = np.linalg.norm(actual)
    norm_predicted = np.linalg.norm(predicted)
    metrics['Cosine_Sim'] = dot_product / (norm_actual * norm_predicted) if norm_actual * norm_predicted != 0 else 0
    
    return metrics

def calculate_structural_metrics(actual, predicted, wavelengths):
    """Calculate structural and perceptual metrics"""
    metrics = {}
    
    # SSIM (treating spectra as 1D signals, reshape to 2D for skimage)
    # Normalize to [0, 1] for SSIM
    actual_norm = (actual - np.min(actual)) / (np.max(actual) - np.min(actual) + 1e-10)
    predicted_norm = (predicted - np.min(predicted)) / (np.max(predicted) - np.min(predicted) + 1e-10)
    
    # Reshape to 2D array (treat as single row image with many columns)
    # Use a small window size appropriate for 1D signal
    window_size = 7
    if len(actual) >= window_size:
        # Pad to make it 2D-like
        actual_2d = np.tile(actual_norm, (window_size, 1))
        predicted_2d = np.tile(predicted_norm, (window_size, 1))
        metrics['SSIM'] = ssim(actual_2d, predicted_2d, win_size=window_size, data_range=1.0)
    else:
        # If signal is too short, use correlation as proxy for SSIM
        metrics['SSIM'] = np.corrcoef(actual_norm, predicted_norm)[0, 1]
    
    # Multi-scale SSIM (simplified for 1D)
    scales = [1, 2, 4]
    ms_ssim_values = []
    for scale in scales:
        if len(actual) >= scale * window_size:
            downsampled_actual = signal.resample(actual_norm, len(actual) // scale)
            downsampled_predicted = signal.resample(predicted_norm, len(predicted) // scale)
            if len(downsampled_actual) >= window_size:
                actual_2d = np.tile(downsampled_actual, (window_size, 1))
                predicted_2d = np.tile(downsampled_predicted, (window_size, 1))
                ms_ssim_values.append(ssim(actual_2d, predicted_2d, win_size=window_size, data_range=1.0))
    metrics['MS_SSIM'] = np.mean(ms_ssim_values) if ms_ssim_values else metrics['SSIM']
    
    # Spectral Angle Mapper (SAM) in radians
    dot_product = np.dot(actual, predicted)
    norm_actual = np.linalg.norm(actual)
    norm_predicted = np.linalg.norm(predicted)
    cos_angle = dot_product / (norm_actual * norm_predicted) if norm_actual * norm_predicted != 0 else 1
    cos_angle = np.clip(cos_angle, -1, 1)  # Ensure within valid range for arccos
    metrics['SAM_radians'] = np.arccos(cos_angle)
    
    return metrics

def calculate_distribution_metrics(actual, predicted):
    """Calculate distribution-based metrics"""
    metrics = {}
    
    # Wasserstein distance (Earth Mover's Distance)
    metrics['Wasserstein'] = wasserstein_distance(actual, predicted)
    
    # Normalize to probability distributions for KL and JS divergence
    actual_prob = actual - np.min(actual) + 1e-10
    actual_prob = actual_prob / np.sum(actual_prob)
    predicted_prob = predicted - np.min(predicted) + 1e-10
    predicted_prob = predicted_prob / np.sum(predicted_prob)
    
    # KL Divergence
    kl_div = stats.entropy(actual_prob, predicted_prob)
    metrics['KL_Divergence'] = kl_div if not np.isinf(kl_div) else np.nan
    
    # Jensen-Shannon Distance
    m = 0.5 * (actual_prob + predicted_prob)
    js_div = 0.5 * stats.entropy(actual_prob, m) + 0.5 * stats.entropy(predicted_prob, m)
    metrics['JS_Distance'] = np.sqrt(js_div)  # JS distance is sqrt of JS divergence
    
    return metrics

def calculate_spectral_metrics(actual, predicted, wavelengths):
    """Calculate spectral-specific metrics"""
    metrics = {}
    
    # Peak wavelength and absorbance
    actual_peak_idx = np.argmax(actual)
    predicted_peak_idx = np.argmax(predicted)
    
    metrics['Peak_Lambda_Error_nm'] = abs(wavelengths[actual_peak_idx] - wavelengths[predicted_peak_idx])
    metrics['Peak_Abs_Error'] = abs(actual[actual_peak_idx] - predicted[predicted_peak_idx])
    
    # FWHM (Full Width at Half Maximum)
    def calculate_fwhm(spectrum, wavelengths):
        peak_idx = np.argmax(spectrum)
        peak_value = spectrum[peak_idx]
        half_max = peak_value / 2
        
        # Find indices where spectrum crosses half maximum
        indices = np.where(spectrum >= half_max)[0]
        if len(indices) > 0:
            return wavelengths[indices[-1]] - wavelengths[indices[0]]
        return 0
    
    actual_fwhm = calculate_fwhm(actual, wavelengths)
    predicted_fwhm = calculate_fwhm(predicted, wavelengths)
    metrics['FWHM_Diff'] = abs(actual_fwhm - predicted_fwhm)
    
    # Integrated area difference (using trapezoidal rule)
    actual_area = np.trapz(actual, wavelengths)
    predicted_area = np.trapz(predicted, wavelengths)
    metrics['Area_Diff'] = abs(actual_area - predicted_area)
    
    return metrics

def calculate_shape_metrics(actual, predicted, wavelengths):
    """Calculate shape-preserving metrics"""
    metrics = {}
    
    # Dynamic Time Warping Distance (simplified version)
    # Using Euclidean distance between aligned points
    n = len(actual)
    dtw_matrix = np.full((n, n), np.inf)
    dtw_matrix[0, 0] = abs(actual[0] - predicted[0])
    
    for i in range(1, n):
        dtw_matrix[i, 0] = dtw_matrix[i-1, 0] + abs(actual[i] - predicted[0])
    for j in range(1, n):
        dtw_matrix[0, j] = dtw_matrix[0, j-1] + abs(actual[0] - predicted[j])
    
    for i in range(1, n):
        for j in range(1, n):
            cost = abs(actual[i] - predicted[j])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
    
    metrics['DTW_Distance'] = dtw_matrix[n-1, n-1] / n  # Normalized by length
    
    # Fréchet Distance (using Hausdorff as approximation)
    # Create 2D curves with wavelength as x-coordinate
    curve_actual = np.column_stack((wavelengths, actual))
    curve_predicted = np.column_stack((wavelengths, predicted))
    
    # Calculate directed Hausdorff distances in both directions
    d_forward = directed_hausdorff(curve_actual, curve_predicted)[0]
    d_backward = directed_hausdorff(curve_predicted, curve_actual)[0]
    metrics['Frechet_Dist'] = max(d_forward, d_backward)
    
    # Derivative MSE (first derivative comparison)
    actual_derivative = np.gradient(actual, wavelengths)
    predicted_derivative = np.gradient(predicted, wavelengths)
    metrics['Derivative_MSE'] = np.mean((actual_derivative - predicted_derivative) ** 2)
    
    return metrics

def calculate_frequency_metrics(actual, predicted):
    """Calculate frequency domain metrics"""
    metrics = {}
    
    # FFT correlation
    fft_actual = np.fft.fft(actual)
    fft_predicted = np.fft.fft(predicted)
    
    # Magnitude correlation in frequency domain
    mag_actual = np.abs(fft_actual)
    mag_predicted = np.abs(fft_predicted)
    metrics['FFT_Correlation'] = np.corrcoef(mag_actual, mag_predicted)[0, 1]
    
    # Power spectral density ratio
    psd_actual = np.abs(fft_actual) ** 2
    psd_predicted = np.abs(fft_predicted) ** 2
    
    # Calculate ratio of total power
    total_power_actual = np.sum(psd_actual)
    total_power_predicted = np.sum(psd_predicted)
    metrics['Power_Ratio'] = total_power_predicted / total_power_actual if total_power_actual != 0 else np.nan
    
    return metrics

def calculate_all_metrics(actual, predicted, wavelengths):
    """Calculate all metrics for a single holdout concentration"""
    all_metrics = {}
    
    # Combine all metric categories
    all_metrics.update(calculate_basic_metrics(actual, predicted))
    all_metrics.update(calculate_correlation_metrics(actual, predicted))
    all_metrics.update(calculate_structural_metrics(actual, predicted, wavelengths))
    all_metrics.update(calculate_distribution_metrics(actual, predicted))
    all_metrics.update(calculate_spectral_metrics(actual, predicted, wavelengths))
    all_metrics.update(calculate_shape_metrics(actual, predicted, wavelengths))
    all_metrics.update(calculate_frequency_metrics(actual, predicted))
    
    return all_metrics

def main():
    """Main function to calculate and save all validation metrics"""
    
    # Load data
    filepath = '/Users/aditya/CodingProjects/STS_September/0.30MB_AuNP_As.csv'
    print("Loading spectral data...")
    wavelengths, concentrations, absorbance_matrix = load_spectral_data(filepath)
    
    print(f"Calculating comprehensive metrics for {len(concentrations)} holdout validations...")
    print("="*80)
    
    # Store results
    results = []
    
    # Calculate metrics for each holdout concentration
    for idx, conc in enumerate(concentrations):
        print(f"Processing holdout concentration: {conc:.0f} ppb")
        
        # Get actual and interpolated spectra
        actual = absorbance_matrix[:, idx]
        interpolated = interpolate_holdout_concentration(wavelengths, concentrations, 
                                                        absorbance_matrix, idx)
        
        # Calculate all metrics
        metrics = calculate_all_metrics(actual, interpolated, wavelengths)
        metrics['Concentration_ppb'] = conc
        
        # Add to results
        results.append(metrics)
        
        # Print key metrics
        print(f"  RMSE: {metrics['RMSE']:.4f}, Pearson R: {metrics['Pearson_R']:.4f}, "
              f"SSIM: {metrics['SSIM']:.4f}, SAM: {metrics['SAM_radians']:.4f} rad")
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Reorder columns with Concentration first
    cols = ['Concentration_ppb'] + [col for col in df_results.columns if col != 'Concentration_ppb']
    df_results = df_results[cols]
    
    # Calculate summary statistics
    mean_row = df_results.mean().to_dict()
    mean_row['Concentration_ppb'] = 'Mean'
    
    std_row = df_results.std().to_dict()
    std_row['Concentration_ppb'] = 'Std'
    
    # Append summary rows
    df_results = pd.concat([df_results, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    
    # Save to CSV
    output_file = 'spectral_validation_metrics.csv'
    df_results.to_csv(output_file, index=False, float_format='%.6f')
    print(f"\n✓ Metrics saved to '{output_file}'")
    
    # Print summary table
    print("\n" + "="*80)
    print("VALIDATION METRICS SUMMARY")
    print("="*80)
    
    # Select key metrics for display
    key_metrics = ['Concentration_ppb', 'RMSE', 'MAE', 'Pearson_R', 'Cosine_Sim', 
                   'SSIM', 'SAM_radians', 'Wasserstein', 'Peak_Lambda_Error_nm']
    
    # Format and display
    print(df_results[key_metrics].to_string(index=False, float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else str(x)))
    
    print("\n" + "="*80)
    print("METRIC INTERPRETATIONS:")
    print("  • Lower is better: RMSE, MAE, MAPE, Max_Error, SAM, Wasserstein, KL_Div, JS_Dist, DTW, Fréchet")
    print("  • Higher is better: Pearson_R, Spearman_R, Cosine_Sim, R2_Score, SSIM, MS_SSIM, FFT_Correlation")
    print("  • Context-dependent: Power_Ratio (should be close to 1)")
    print("="*80)
    
    # Create correlation heatmap
    create_metric_visualization(df_results[df_results['Concentration_ppb'] != 'Mean'][df_results['Concentration_ppb'] != 'Std'])
    
    return df_results

def create_metric_visualization(df_results):
    """Create interactive visualization of metrics"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    
    # Select numeric columns only
    numeric_cols = [col for col in df_results.columns if col != 'Concentration_ppb']
    df_numeric = df_results[numeric_cols]
    
    # Normalize metrics to 0-1 scale for comparison
    df_normalized = df_numeric.copy()
    for col in df_normalized.columns:
        col_min = df_normalized[col].min()
        col_max = df_normalized[col].max()
        if col_max != col_min:
            df_normalized[col] = (df_normalized[col] - col_min) / (col_max - col_min)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Normalized Metrics Heatmap', 'Metric Correlations',
                       'Error Metrics by Concentration', 'Similarity Metrics by Concentration'),
        specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}],
               [{'type': 'scatter'}, {'type': 'scatter'}]]
    )
    
    # 1. Normalized metrics heatmap
    concentrations = df_results['Concentration_ppb'].values
    z_data = df_normalized.values
    
    heatmap1 = go.Heatmap(
        z=z_data,
        x=df_normalized.columns,
        y=[f'{c} ppb' for c in concentrations],
        colorscale='RdYlGn_r',
        showscale=True,
        colorbar=dict(title='Normalized<br>Value', x=0.45, len=0.4)
    )
    fig.add_trace(heatmap1, row=1, col=1)
    
    # 2. Correlation matrix
    corr_matrix = df_numeric.corr()
    heatmap2 = go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        showscale=True,
        colorbar=dict(title='Correlation', x=1.0, len=0.4)
    )
    fig.add_trace(heatmap2, row=1, col=2)
    
    # 3. Error metrics line plot
    error_metrics = ['RMSE', 'MAE', 'Max_Error']
    for metric in error_metrics:
        fig.add_trace(
            go.Scatter(
                x=concentrations,
                y=df_results[metric].values,
                mode='lines+markers',
                name=metric,
                line=dict(width=2),
                marker=dict(size=8)
            ),
            row=2, col=1
        )
    
    # 4. Similarity metrics line plot
    similarity_metrics = ['Pearson_R', 'Cosine_Sim', 'SSIM']
    for metric in similarity_metrics:
        fig.add_trace(
            go.Scatter(
                x=concentrations,
                y=df_results[metric].values,
                mode='lines+markers',
                name=metric,
                line=dict(width=2),
                marker=dict(size=8)
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_xaxes(title_text="Metrics", row=1, col=1, tickangle=45)
    fig.update_xaxes(title_text="Metrics", row=1, col=2, tickangle=45)
    fig.update_xaxes(title_text="Concentration (ppb)", row=2, col=1)
    fig.update_xaxes(title_text="Concentration (ppb)", row=2, col=2)
    
    fig.update_yaxes(title_text="Concentration", row=1, col=1)
    fig.update_yaxes(title_text="Metrics", row=1, col=2)
    fig.update_yaxes(title_text="Error Value", row=2, col=1)
    fig.update_yaxes(title_text="Similarity Score", row=2, col=2)
    
    fig.update_layout(
        title={
            'text': 'Comprehensive Spectral Validation Metrics Analysis',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        height=900,
        width=1400,
        showlegend=True,
        paper_bgcolor='#f8f9fa'
    )
    
    # Save visualization
    output_file = 'spectral_metrics_analysis.html'
    fig.write_html(output_file, include_plotlyjs='cdn')
    print(f"✓ Visualization saved to '{output_file}'")

if __name__ == "__main__":
    main()