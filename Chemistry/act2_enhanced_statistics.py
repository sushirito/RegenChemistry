#!/usr/bin/env python3
"""
Act II: Enhanced Statistical Analysis
Panel A: Dialysis kinetics (verbatim)
Panel B: Enhanced redshift analysis with statistics and quality metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Professional styling with more space
plt.style.use('default')
plt.rcParams.update({
    'figure.dpi': 300,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'lines.linewidth': 2.5,
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8,
})

# Professional colors
COLORS = {
    'dialysis': '#1f77b4',      # Blue
    'centrifuge': '#ff7f0e',    # Orange
    'citrate': '#ffcc00',       # Yellow for citrate baseline
    'accent': '#2ca02c',        # Green
    'background': '#f8f9fa',    # Light background
    'table_header': '#e6f3ff',  # Light blue for table headers
}

print("="*80)
print("ACT II: ENHANCED STATISTICAL ANALYSIS")
print("="*80)

# =============================================================================
# PANEL A: EXACT DIALYSIS KINETICS (VERBATIM FROM REFERENCE)
# =============================================================================

print("\nPanel A: Using exact experimental data...")

# Exact data from dialysis_plots.py
V_inside_mL = 10.0
CITRATE_ONLY_M0_mg = 0.375  # 1.0 mg citrate × 0.375 C fraction

data = [
    {"run": 1, "t_hr": 16.0, "C_meas_mg_per_L": 1.04, "C_adj_mg_per_L": 35.0},
    {"run": 2, "t_hr": 22.0, "C_meas_mg_per_L": 0.58, "C_adj_mg_per_L": 18.5},
    {"run": 3, "t_hr": 40.0, "C_meas_mg_per_L": 0.52, "C_adj_mg_per_L": 16.4},
]

df_kinetics = pd.DataFrame(data)

# Calculate masses exactly as in dialysis_plots.py
df_kinetics["V_bath_mL"] = (df_kinetics["C_adj_mg_per_L"] / df_kinetics["C_meas_mg_per_L"]) * V_inside_mL
df_kinetics["V_bath_L"] = df_kinetics["V_bath_mL"] / 1000.0
df_kinetics["mass_mg"] = df_kinetics["C_meas_mg_per_L"] * df_kinetics["V_bath_L"]
df_kinetics = df_kinetics.sort_values("t_hr").reset_index(drop=True)
df_kinetics["mass_cum_mg"] = df_kinetics["mass_mg"].cumsum()

# Add t=0 point
time_points = np.array([0] + df_kinetics["t_hr"].tolist())
mass_cumulative = np.array([0] + df_kinetics["mass_cum_mg"].tolist())

# First-order kinetics model
def M_cum_model(t, M0_hat, k):
    return M0_hat * (1.0 - np.exp(-k * t))

# Fit model
t_data = time_points[1:]
y_data = mass_cumulative[1:]

popt, pcov = curve_fit(M_cum_model, t_data, y_data, p0=[0.8, 0.03], maxfev=20000)
M0_hat, k_hat = popt
perr = np.sqrt(np.diag(pcov))
t_half = np.log(2.0)/k_hat if k_hat > 0 else np.nan

# Generate confidence bands (exact method from dialysis_plots.py)
np.random.seed(0)
try:
    draws = np.random.multivariate_normal(mean=popt, cov=pcov, size=1000)
except np.linalg.LinAlgError:
    draws = np.column_stack([
        np.random.normal(M0_hat, np.sqrt(pcov[0,0]), 1000),
        np.random.normal(k_hat, np.sqrt(pcov[1,1]), 1000),
    ])

tgrid = np.linspace(0, float(df_kinetics["t_hr"].max()), 300)
Y = np.array([M_cum_model(tgrid, *theta) for theta in draws])
ci_lo = np.percentile(Y, 2.5, axis=0)
ci_hi = np.percentile(Y, 97.5, axis=0)
yfit = M_cum_model(tgrid, *popt)

# =============================================================================
# ENHANCED SPECTRAL ANALYSIS WITH COMPREHENSIVE METRICS
# =============================================================================

print("\nPanel B: Loading spectral data for comprehensive analysis...")

df_spectra = pd.read_csv('/Users/aditya/CodingProjects/Chemistry/data/clean_data/UVScans_CleanedAbsorbance.csv')

pairs = [
    ('0.115MB_AuNP_MQW', '0.115MB_cenAuNP_MQW', '0.115 MB MQW'),
    ('0.30MB_AuNP_MQW', '0.30MB_cenAuNP_MQW', '0.30 MB MQW'),
    ('0.115MB_AuNP_As30', '0.115MB_cenAuNP_As30', '0.115 MB As30'),
    ('0.30MB_AuNP_As30_1', '0.30MB_cenAuNP_As30', '0.30 MB As30'),
]

wavelengths = df_spectra['Wavelength'].values

def find_peak_position_precise(spectrum, wavelengths, search_range=(510, 540)):
    """Find LSPR peak position with high precision"""
    mask = (wavelengths >= search_range[0]) & (wavelengths <= search_range[1])
    if not np.any(mask):
        return np.nan

    region_wl = wavelengths[mask]
    region_abs = spectrum[mask]
    peak_idx = np.argmax(region_abs)

    # Parabolic interpolation for sub-nm precision
    if 1 <= peak_idx <= len(region_abs) - 2:
        y1, y2, y3 = region_abs[peak_idx-1:peak_idx+2]
        x1, x2, x3 = region_wl[peak_idx-1:peak_idx+2]

        denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
        if abs(denom) > 1e-10:
            a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
            b = (x3*x3 * (y1 - y2) + x2*x2 * (y3 - y1) + x1*x1 * (y2 - y3)) / denom

            if abs(a) > 1e-10:
                peak_pos = -b / (2 * a)
                if search_range[0] <= peak_pos <= search_range[1]:
                    return peak_pos

    return region_wl[peak_idx]

def calculate_spectral_quality_metrics(spectrum, wavelengths):
    """Calculate comprehensive spectral quality metrics"""
    metrics = {}

    # 1. Peak position
    metrics['peak_position'] = find_peak_position_precise(spectrum, wavelengths)

    # 2. Peak height (520nm region)
    peak_region = (wavelengths >= 515) & (wavelengths <= 525)
    if np.any(peak_region):
        metrics['peak_height'] = np.max(spectrum[peak_region])
    else:
        metrics['peak_height'] = np.nan

    # 3. FWHM calculation
    peak_window = (wavelengths >= 480) & (wavelengths <= 560)
    if np.any(peak_window):
        w_win = wavelengths[peak_window]
        s_win = spectrum[peak_window]
        peak_idx = np.argmax(s_win)
        peak_val = s_win[peak_idx]
        half_max = peak_val / 2

        # Find FWHM
        left_half = s_win[:peak_idx+1][::-1]
        right_half = s_win[peak_idx:]

        try:
            left_cross = np.interp(half_max, left_half, w_win[:peak_idx+1][::-1])
            right_cross = np.interp(half_max, right_half, w_win[peak_idx:])
            metrics['fwhm'] = right_cross - left_cross
        except:
            metrics['fwhm'] = np.nan
    else:
        metrics['fwhm'] = np.nan

    # 4. Red tail absorption (aggregation proxy)
    red_tail_region = (wavelengths >= 650) & (wavelengths <= 700)
    if np.any(red_tail_region):
        metrics['red_tail'] = np.mean(spectrum[red_tail_region])
    else:
        metrics['red_tail'] = np.nan

    # 5. Baseline noise (700-720nm)
    noise_region = (wavelengths >= 700) & (wavelengths <= 720)
    if np.any(noise_region):
        metrics['baseline_noise'] = np.std(spectrum[noise_region])
    else:
        metrics['baseline_noise'] = np.nan

    # 6. Peak asymmetry (red vs blue side width)
    if not np.isnan(metrics['peak_position']):
        peak_pos = metrics['peak_position']
        peak_mask = (wavelengths >= peak_pos-20) & (wavelengths <= peak_pos+20)
        if np.any(peak_mask):
            w_asym = wavelengths[peak_mask]
            s_asym = spectrum[peak_mask]
            peak_idx_asym = np.argmax(s_asym)

            blue_width = peak_pos - w_asym[0]
            red_width = w_asym[-1] - peak_pos
            metrics['asymmetry'] = red_width / blue_width if blue_width > 0 else np.nan
        else:
            metrics['asymmetry'] = np.nan
    else:
        metrics['asymmetry'] = np.nan

    return metrics

# Calculate comprehensive metrics for all pairs
print("Calculating comprehensive spectral metrics...")

baseline_peak = 520.0  # Reference peak position
all_metrics = []

for dial_col, cent_col, label in pairs:
    dial_spectrum = df_spectra[dial_col].values
    cent_spectrum = df_spectra[cent_col].values

    dial_metrics = calculate_spectral_quality_metrics(dial_spectrum, wavelengths)
    cent_metrics = calculate_spectral_quality_metrics(cent_spectrum, wavelengths)

    # Calculate redshift from baseline
    dial_redshift = dial_metrics['peak_position'] - baseline_peak
    cent_redshift = cent_metrics['peak_position'] - baseline_peak

    result = {
        'sample': label,
        'dialysis_peak': dial_metrics['peak_position'],
        'centrifuge_peak': cent_metrics['peak_position'],
        'dialysis_redshift': dial_redshift,
        'centrifuge_redshift': cent_redshift,
        'dial_peak_height': dial_metrics['peak_height'],
        'cent_peak_height': cent_metrics['peak_height'],
        'dial_fwhm': dial_metrics['fwhm'],
        'cent_fwhm': cent_metrics['fwhm'],
        'dial_red_tail': dial_metrics['red_tail'],
        'cent_red_tail': cent_metrics['red_tail'],
        'dial_noise': dial_metrics['baseline_noise'],
        'cent_noise': cent_metrics['baseline_noise'],
        'dial_asymmetry': dial_metrics['asymmetry'],
        'cent_asymmetry': cent_metrics['asymmetry'],
    }

    all_metrics.append(result)

metrics_df = pd.DataFrame(all_metrics)

# Calculate statistical comparisons
def calculate_method_statistics():
    """Calculate statistical comparisons between methods"""
    stats_results = {}

    metric_pairs = [
        ('redshift', 'dialysis_redshift', 'centrifuge_redshift'),
        ('peak_height', 'dial_peak_height', 'cent_peak_height'),
        ('fwhm', 'dial_fwhm', 'cent_fwhm'),
        ('red_tail', 'dial_red_tail', 'cent_red_tail'),
        ('noise', 'dial_noise', 'cent_noise'),
        ('asymmetry', 'dial_asymmetry', 'cent_asymmetry'),
    ]

    for metric_name, dial_col, cent_col in metric_pairs:
        dial_vals = metrics_df[dial_col].dropna().values
        cent_vals = metrics_df[cent_col].dropna().values

        if len(dial_vals) > 0 and len(cent_vals) > 0:
            # Paired t-test
            if len(dial_vals) == len(cent_vals):
                t_stat, p_val = stats.ttest_rel(dial_vals, cent_vals)
            else:
                t_stat, p_val = stats.ttest_ind(dial_vals, cent_vals)

            # Calculate means and standard errors
            dial_mean = np.mean(dial_vals)
            cent_mean = np.mean(cent_vals)
            dial_sem = stats.sem(dial_vals)
            cent_sem = stats.sem(cent_vals)

            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(dial_vals)-1)*np.var(dial_vals, ddof=1) +
                                 (len(cent_vals)-1)*np.var(cent_vals, ddof=1)) /
                                (len(dial_vals) + len(cent_vals) - 2))
            cohens_d = (dial_mean - cent_mean) / pooled_std if pooled_std > 0 else 0

            stats_results[metric_name] = {
                'dialysis_mean': dial_mean,
                'dialysis_sem': dial_sem,
                'centrifuge_mean': cent_mean,
                'centrifuge_sem': cent_sem,
                'p_value': p_val,
                'cohens_d': cohens_d,
                'significant': p_val < 0.05
            }

    return stats_results

statistical_results = calculate_method_statistics()

print("Statistical analysis complete!")

# =============================================================================
# CREATE ENHANCED FIGURE WITH PROPER SPACING
# =============================================================================

print("Creating enhanced figure with statistics...")

# Create figure with more space for tables
fig = plt.figure(figsize=(18, 12))

# Define grid: left side for plots, right side for tables
gs_main = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1.2, 1], wspace=0.3)
gs_left = gridspec.GridSpecFromSubplotSpec(1, 1, gs_main[0])
gs_center = gridspec.GridSpecFromSubplotSpec(1, 1, gs_main[1])
gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, gs_main[2], hspace=0.4)

# =============================================================================
# PANEL A: DIALYSIS KINETICS (VERBATIM)
# =============================================================================

ax1 = fig.add_subplot(gs_left[0])

# Plot experimental data points
ax1.scatter(time_points, mass_cumulative, s=80, color=COLORS['dialysis'],
           zorder=5, edgecolors='white', linewidths=1.5, label='Experimental data')

# Plot fitted curve
ax1.plot(tgrid, yfit, color=COLORS['dialysis'], linewidth=2.5,
         label=f"First-order fit (t½ = {t_half:.1f} h)")

# Add 95% confidence interval
ax1.fill_between(tgrid, ci_lo, ci_hi, color=COLORS['dialysis'], alpha=0.2,
                 label='95% CI')

# Add citrate-only baseline
ax1.axhline(CITRATE_ONLY_M0_mg, color=COLORS['citrate'], linestyle='--',
           linewidth=2, label='Citrate-only (theoretical)')

# Formatting
ax1.set_xlabel('Time (hours)', fontweight='bold')
ax1.set_ylabel('Cumulative Carbon Removed (mg)', fontweight='bold')
ax1.set_title('A. Kinetics of Purification', fontsize=15, fontweight='bold', pad=15)
ax1.set_xlim(0, 42)
ax1.set_ylim(0, max(mass_cumulative) * 1.2)

# Parameter text box
param_text = f'M₀ = {M0_hat:.2f} ± {perr[0]:.2f} mg\nk = {k_hat:.3f} ± {perr[1]:.3f} h⁻¹\nt½ = {t_half:.1f} h'
ax1.text(0.02, 0.98, param_text, transform=ax1.transAxes,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9),
         fontsize=10, ha='left', va='top', fontfamily='monospace')

ax1.legend(loc='lower right', frameon=True, fancybox=True)

# =============================================================================
# PANEL B: ENHANCED REDSHIFT ANALYSIS WITH ERROR BARS
# =============================================================================

ax2 = fig.add_subplot(gs_center[0])

# Calculate standard errors for error bars (using bootstrap approach)
n_bootstrap = 100
np.random.seed(42)

dial_redshift_errors = []
cent_redshift_errors = []

for _, row in metrics_df.iterrows():
    sample = row['sample']

    # Find the corresponding spectra
    dial_col = None
    cent_col = None
    for d_col, c_col, label in pairs:
        if label == sample:
            dial_col = d_col
            cent_col = c_col
            break

    if dial_col is not None:
        dial_spec = df_spectra[dial_col].values
        cent_spec = df_spectra[cent_col].values

        # Bootstrap peak position uncertainty
        dial_peaks = []
        cent_peaks = []

        for _ in range(n_bootstrap):
            # Add small amount of noise
            noise_level = 0.001  # 0.1% noise
            dial_noisy = dial_spec + np.random.normal(0, noise_level, len(dial_spec))
            cent_noisy = cent_spec + np.random.normal(0, noise_level, len(cent_spec))

            dial_peak = find_peak_position_precise(dial_noisy, wavelengths)
            cent_peak = find_peak_position_precise(cent_noisy, wavelengths)

            if not np.isnan(dial_peak):
                dial_peaks.append(dial_peak - baseline_peak)
            if not np.isnan(cent_peak):
                cent_peaks.append(cent_peak - baseline_peak)

        dial_redshift_errors.append(np.std(dial_peaks) if dial_peaks else 0.1)
        cent_redshift_errors.append(np.std(cent_peaks) if cent_peaks else 0.1)
    else:
        dial_redshift_errors.append(0.1)
        cent_redshift_errors.append(0.1)

# Create grouped bar plot with error bars
x_pos = np.arange(len(metrics_df))
width = 0.35

bars1 = ax2.bar(x_pos - width/2, metrics_df['dialysis_redshift'], width,
                yerr=dial_redshift_errors, capsize=4,
                label='Dialysis', color=COLORS['dialysis'], alpha=0.8,
                edgecolor='black', linewidth=1)
bars2 = ax2.bar(x_pos + width/2, metrics_df['centrifuge_redshift'], width,
                yerr=cent_redshift_errors, capsize=4,
                label='Centrifuge', color=COLORS['centrifuge'], alpha=0.8,
                edgecolor='black', linewidth=1)

# Add value labels on bars
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    h1, h2 = bar1.get_height(), bar2.get_height()

    # Adjust label position based on error bar
    err1 = dial_redshift_errors[i]
    err2 = cent_redshift_errors[i]

    ax2.text(bar1.get_x() + bar1.get_width()/2, h1 + err1 + 0.3,
             f'{h1:+.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.text(bar2.get_x() + bar2.get_width()/2, h2 + err2 + 0.3,
             f'{h2:+.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Reference line at zero
ax2.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)

# Statistics summary
redshift_stats = statistical_results.get('redshift', {})
stats_text = f'p-value: {redshift_stats.get("p_value", 0):.3f}\n'
stats_text += f'Effect size: {redshift_stats.get("cohens_d", 0):.2f}'

ax2.text(0.98, 0.02, stats_text, transform=ax2.transAxes,
         bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['background'], alpha=0.9),
         fontsize=10, ha='right', va='bottom', fontweight='bold')

# Formatting
ax2.set_xlabel('Sample Pair', fontweight='bold')
ax2.set_ylabel('LSPR Peak Shift from 520nm (nm)', fontweight='bold')
ax2.set_title('B. Method-Induced Spectral Shifts', fontsize=15, fontweight='bold', pad=15)
ax2.set_xticks(x_pos)
ax2.set_xticklabels([s.replace(' ', '\n') for s in metrics_df['sample']], fontsize=9)
ax2.legend(frameon=True, fancybox=True)

# =============================================================================
# PANEL C: STATISTICAL SUMMARY TABLE
# =============================================================================

ax3 = fig.add_subplot(gs_right[0])
ax3.axis('off')

# Create statistical summary table
table_data = []
headers = ['Metric', 'Dialysis\n(Mean±SEM)', 'Centrifuge\n(Mean±SEM)', 'p-value', 'Significant']

metric_display_names = {
    'redshift': 'Peak Shift (nm)',
    'peak_height': 'Peak Height',
    'fwhm': 'FWHM (nm)',
    'red_tail': 'Red Tail',
    'noise': 'Baseline Noise',
    'asymmetry': 'Peak Asymmetry'
}

for metric_name, display_name in metric_display_names.items():
    if metric_name in statistical_results:
        stats = statistical_results[metric_name]

        dial_str = f"{stats['dialysis_mean']:.3f}±{stats['dialysis_sem']:.3f}"
        cent_str = f"{stats['centrifuge_mean']:.3f}±{stats['centrifuge_sem']:.3f}"
        p_str = f"{stats['p_value']:.3f}"
        sig_str = "Yes" if stats['significant'] else "No"

        table_data.append([display_name, dial_str, cent_str, p_str, sig_str])

# Create table
table = ax3.table(cellText=table_data, colLabels=headers,
                 cellLoc='center', loc='center')

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 2)

# Style the table
for i in range(len(headers)):
    table[(0, i)].set_facecolor(COLORS['table_header'])
    table[(0, i)].set_text_props(weight='bold')

# Color code significance
for i, row in enumerate(table_data, 1):
    if row[4] == 'Yes':  # Significant
        table[(i, 4)].set_facecolor('#d4edda')  # Light green
    else:
        table[(i, 4)].set_facecolor('#f8d7da')  # Light red

ax3.set_title('C. Statistical Comparison', fontsize=13, fontweight='bold', pad=20)

# =============================================================================
# PANEL D: QUALITY METRICS COMPARISON
# =============================================================================

ax4 = fig.add_subplot(gs_right[1])
ax4.axis('off')

# Create quality metrics comparison table
quality_data = []
quality_headers = ['Sample', 'Method', 'Peak Pos.\n(nm)', 'Height', 'FWHM\n(nm)', 'Red Tail']

for _, row in metrics_df.iterrows():
    # Dialysis row
    quality_data.append([
        row['sample'], 'Dialysis',
        f"{row['dialysis_peak']:.1f}",
        f"{row['dial_peak_height']:.3f}",
        f"{row['dial_fwhm']:.1f}" if not np.isnan(row['dial_fwhm']) else 'N/A',
        f"{row['dial_red_tail']:.3f}"
    ])

    # Centrifuge row
    quality_data.append([
        '', 'Centrifuge',
        f"{row['centrifuge_peak']:.1f}",
        f"{row['cent_peak_height']:.3f}",
        f"{row['cent_fwhm']:.1f}" if not np.isnan(row['cent_fwhm']) else 'N/A',
        f"{row['cent_red_tail']:.3f}"
    ])

# Create quality table
quality_table = ax4.table(cellText=quality_data, colLabels=quality_headers,
                         cellLoc='center', loc='center')

quality_table.auto_set_font_size(False)
quality_table.set_fontsize(7)
quality_table.scale(1, 1.5)

# Style the quality table
for i in range(len(quality_headers)):
    quality_table[(0, i)].set_facecolor(COLORS['table_header'])
    quality_table[(0, i)].set_text_props(weight='bold', size=8)

# Alternate row colors for better readability
for i in range(1, len(quality_data) + 1):
    if i % 4 in [1, 2]:  # First pair of each sample
        color = '#f8f9fa'
    else:  # Second pair
        color = 'white'

    for j in range(len(quality_headers)):
        quality_table[(i, j)].set_facecolor(color)

ax4.set_title('D. Quality Metrics Comparison', fontsize=13, fontweight='bold', pad=20)

# =============================================================================
# FINAL FORMATTING
# =============================================================================

plt.suptitle('Figure 2: Comprehensive Dialysis vs Centrifugation Analysis',
            fontsize=16, fontweight='bold', y=0.95)

# Save outputs
output_dir = Path("/Users/aditya/CodingProjects/Chemistry/figures/act2_enhanced")
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(output_dir / "figure2_enhanced_statistics.png",
           dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(output_dir / "figure2_enhanced_statistics.pdf",
           bbox_inches='tight', facecolor='white')

# Save data
metrics_df.to_csv(output_dir / "comprehensive_spectral_metrics.csv", index=False)

# Save statistical results
stats_summary = pd.DataFrame(statistical_results).T
stats_summary.to_csv(output_dir / "statistical_comparison_results.csv")

print("\n" + "="*80)
print("ENHANCED ANALYSIS COMPLETE!")
print("="*80)
print("Panel A: Dialysis kinetics (verbatim from reference)")
print("Panel B: Enhanced redshift analysis with error bars")
print("Panel C: Statistical summary with p-values")
print("Panel D: Comprehensive quality metrics table")
print(f"\nStatistical Results:")
for metric, stats in statistical_results.items():
    sig_str = "***" if stats['p_value'] < 0.001 else "**" if stats['p_value'] < 0.01 else "*" if stats['p_value'] < 0.05 else "ns"
    print(f"  {metric}: p={stats['p_value']:.3f} {sig_str}, d={stats['cohens_d']:.2f}")

print(f"\nFiles saved to: {output_dir}")
plt.close('all')
print("\n🎯 Enhanced statistical analysis complete!")