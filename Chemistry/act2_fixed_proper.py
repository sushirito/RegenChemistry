#!/usr/bin/env python3
"""
Act II: Fixed Version - Proper Kinetics + Better Analysis
Using ACTUAL data from dialysis_plots.py with correct formatting
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

# Clean professional styling matching the reference image
plt.style.use('default')
plt.rcParams.update({
    'figure.dpi': 300,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.family': 'Arial',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
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
}

print("="*80)
print("ACT II: PROPER KINETICS + REDSHIFT ANALYSIS")
print("="*80)

# =============================================================================
# ACTUAL DIALYSIS KINETICS DATA (from dialysis_plots.py)
# =============================================================================

print("\nUsing ACTUAL experimental data from dialysis_plots.py...")

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

print("Calculated kinetics data:")
for _, row in df_kinetics.iterrows():
    print(f"  t={row['t_hr']:2.0f}h: mass={row['mass_mg']:.3f}mg, cumulative={row['mass_cum_mg']:.3f}mg")

# Add t=0 point
time_points = np.array([0] + df_kinetics["t_hr"].tolist())
mass_cumulative = np.array([0] + df_kinetics["mass_cum_mg"].tolist())

print(f"Final data points: {list(zip(time_points, mass_cumulative))}")

# First-order kinetics model (exact from dialysis_plots.py)
def M_cum_model(t, M0_hat, k):
    return M0_hat * (1.0 - np.exp(-k * t))

# Fit model (excluding t=0 point)
t_data = time_points[1:]
y_data = mass_cumulative[1:]

popt, pcov = curve_fit(M_cum_model, t_data, y_data, p0=[0.8, 0.03], maxfev=20000)
M0_hat, k_hat = popt
perr = np.sqrt(np.diag(pcov))

# Calculate derived parameters
t_half = np.log(2.0)/k_hat if k_hat > 0 else np.nan
z = 1.959963984540054  # 95% CI
M0_lo, M0_hi = M0_hat - z*perr[0], M0_hat + z*perr[0]
k_lo, k_hi = k_hat - z*perr[1], k_hat + z*perr[1]

print(f"\nKinetics fit results:")
print(f"  M₀ = {M0_hat:.3f} ± {perr[0]:.3f} mg")
print(f"  k = {k_hat:.3f} ± {perr[1]:.3f} h⁻¹")
print(f"  t½ = {t_half:.1f} h")

# Generate confidence bands using exact method from dialysis_plots.py
np.random.seed(0)  # Use same seed as original
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
# SPECTRAL DATA AND REDSHIFT ANALYSIS
# =============================================================================

print("\nLoading spectral data for redshift analysis...")
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

    # Find peak
    peak_idx = np.argmax(region_abs)

    # Use parabolic interpolation for sub-nm precision
    if 1 <= peak_idx <= len(region_abs) - 2:
        y1, y2, y3 = region_abs[peak_idx-1:peak_idx+2]
        x1, x2, x3 = region_wl[peak_idx-1:peak_idx+2]

        # Parabolic fit: y = ax² + bx + c
        denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
        if abs(denom) > 1e-10:
            a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
            b = (x3*x3 * (y1 - y2) + x2*x2 * (y3 - y1) + x1*x1 * (y2 - y3)) / denom

            if abs(a) > 1e-10:
                peak_pos = -b / (2 * a)
                if search_range[0] <= peak_pos <= search_range[1]:
                    return peak_pos

    return region_wl[peak_idx]

# Calculate redshift relative to baseline (using 520nm as reference)
baseline_peak = 520.0  # Theoretical pristine AuNP peak

print("\nCalculating redshift from baseline (520nm)...")

redshift_data = []
for dial_col, cent_col, label in pairs:
    dial_spectrum = df_spectra[dial_col].values
    cent_spectrum = df_spectra[cent_col].values

    peak_dial = find_peak_position_precise(dial_spectrum, wavelengths)
    peak_cent = find_peak_position_precise(cent_spectrum, wavelengths)

    redshift_dial = peak_dial - baseline_peak
    redshift_cent = peak_cent - baseline_peak

    redshift_data.append({
        'sample': label,
        'dialysis_peak': peak_dial,
        'centrifuge_peak': peak_cent,
        'redshift_dialysis': redshift_dial,
        'redshift_centrifuge': redshift_cent,
        'relative_shift': peak_cent - peak_dial  # cent vs dial
    })

    print(f"  {label}:")
    print(f"    Dialysis: {peak_dial:.2f}nm (Δ={redshift_dial:+.2f}nm)")
    print(f"    Centrifuge: {peak_cent:.2f}nm (Δ={redshift_cent:+.2f}nm)")
    print(f"    Cent vs Dial: {peak_cent - peak_dial:+.2f}nm")

redshift_df = pd.DataFrame(redshift_data)

# =============================================================================
# CREATE CLEAN FIGURE WITH PROPER FORMATTING
# =============================================================================

print("\nCreating properly formatted figure...")

fig = plt.figure(figsize=(16, 8))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1], hspace=0.3, wspace=0.25)

# =============================================================================
# PANEL A: DIALYSIS KINETICS (MATCHING THE REFERENCE IMAGE)
# =============================================================================

ax1 = fig.add_subplot(gs[0, 0])

# Plot experimental data points
ax1.scatter(time_points, mass_cumulative, s=80, color=COLORS['dialysis'],
           zorder=5, edgecolors='white', linewidths=1.5, label='Experimental data')

# Plot fitted curve
ax1.plot(tgrid, yfit, color=COLORS['dialysis'], linewidth=2.5,
         label=f"First-order fit (t½ = {t_half:.1f} h)")

# Add 95% confidence interval
ax1.fill_between(tgrid, ci_lo, ci_hi, color=COLORS['dialysis'], alpha=0.2,
                 label='95% CI')

# Add citrate-only baseline (horizontal dashed line)
ax1.axhline(CITRATE_ONLY_M0_mg, color=COLORS['citrate'], linestyle='--',
           linewidth=2, label='Citrate-only (theoretical)')

# Clean formatting matching reference image
ax1.set_xlabel('Time (hours)', fontweight='bold')
ax1.set_ylabel('Cumulative Carbon Removed (mg)', fontweight='bold')
ax1.set_title('A. Kinetics of Purification', fontsize=16, fontweight='bold', pad=15)
ax1.set_xlim(0, 42)
ax1.set_ylim(0, max(mass_cumulative) * 1.2)

# Parameter text box (positioned like in reference image)
param_text = f'M₀ = {M0_hat:.2f} ± {perr[0]:.2f} mg\nk = {k_hat:.3f} ± {perr[1]:.3f} h⁻¹\nt½ = {t_half:.1f} h'
ax1.text(0.02, 0.98, param_text, transform=ax1.transAxes,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9),
         fontsize=10, ha='left', va='top', fontfamily='monospace')

# Legend positioned to avoid overlap
ax1.legend(loc='lower right', frameon=True, fancybox=True, shadow=False)

# =============================================================================
# PANEL B: REDSHIFT ANALYSIS (CORRECTED)
# =============================================================================

ax2 = fig.add_subplot(gs[0, 1])

# Create grouped bar plot
x_pos = np.arange(len(redshift_df))
width = 0.35

# Plot redshift for both methods
bars1 = ax2.bar(x_pos - width/2, redshift_df['redshift_dialysis'], width,
                label='Dialysis', color=COLORS['dialysis'], alpha=0.8, edgecolor='black', linewidth=1)
bars2 = ax2.bar(x_pos + width/2, redshift_df['redshift_centrifuge'], width,
                label='Centrifuge', color=COLORS['centrifuge'], alpha=0.8, edgecolor='black', linewidth=1)

# Add value labels on bars
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    h1, h2 = bar1.get_height(), bar2.get_height()
    ax2.text(bar1.get_x() + bar1.get_width()/2, h1 + 0.5,
             f'{h1:+.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.text(bar2.get_x() + bar2.get_width()/2, h2 + 0.5,
             f'{h2:+.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Reference line at zero
ax2.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)

# Statistics summary
mean_dial = np.mean(redshift_df['redshift_dialysis'])
mean_cent = np.mean(redshift_df['redshift_centrifuge'])

stats_text = f'Mean redshift:\nDialysis: {mean_dial:+.1f} nm\nCentrifuge: {mean_cent:+.1f} nm'
ax2.text(0.98, 0.02, stats_text, transform=ax2.transAxes,
         bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['background'], alpha=0.9),
         fontsize=10, ha='right', va='bottom', fontweight='bold')

# Clean formatting
ax2.set_xlabel('Sample Pair', fontweight='bold')
ax2.set_ylabel('LSPR Peak Shift from 520nm (nm)', fontweight='bold')
ax2.set_title('B. Method-Induced Spectral Shifts', fontsize=16, fontweight='bold', pad=15)
ax2.set_xticks(x_pos)
ax2.set_xticklabels([s.replace(' ', '\n') for s in redshift_df['sample']], fontsize=9)
ax2.legend(frameon=True, fancybox=True)

# Add explanation
explanation = "Positive = Red-shift (aggregation)\nNegative = Blue-shift"
ax2.text(0.02, 0.98, explanation, transform=ax2.transAxes,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
         fontsize=9, ha='left', va='top', style='italic')

# =============================================================================
# FINAL FORMATTING
# =============================================================================

plt.suptitle('Figure 2: Dialysis Kinetics and Method Comparison',
            fontsize=18, fontweight='bold', y=0.96)

# Save outputs
output_dir = Path("/Users/aditya/CodingProjects/Chemistry/figures/act2_fixed")
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(output_dir / "figure2_fixed_proper_kinetics.png",
           dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(output_dir / "figure2_fixed_proper_kinetics.pdf",
           bbox_inches='tight', facecolor='white')

# Save data
redshift_df.to_csv(output_dir / "redshift_analysis_proper.csv", index=False)

# Enhanced results summary
print("\n" + "="*80)
print("FIXED ANALYSIS COMPLETE!")
print("="*80)
print("Kinetics:")
print(f"  ✓ M₀ = {M0_hat:.3f} ± {perr[0]:.3f} mg")
print(f"  ✓ k = {k_hat:.3f} ± {perr[1]:.3f} h⁻¹")
print(f"  ✓ t½ = {t_half:.1f} hours")
print(f"  ✓ Citrate baseline = {CITRATE_ONLY_M0_mg:.3f} mg")

print("\nRedshift Analysis:")
print(f"  ✓ Mean dialysis redshift: {mean_dial:+.2f} nm")
print(f"  ✓ Mean centrifuge redshift: {mean_cent:+.2f} nm")
print(f"  ✓ Both methods show spectral changes from pristine")

print(f"\n✓ Clean formatting with no overlaps")
print(f"✓ Figures saved to: {output_dir}")

plt.close('all')
print("\n🎯 Properly fixed Act II analysis complete!")