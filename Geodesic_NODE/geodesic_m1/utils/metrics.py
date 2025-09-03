"""
Comprehensive metrics computation for spectral validation
Includes 20+ metrics across multiple categories
"""

import numpy as np
from scipy import stats, signal
from scipy.interpolate import interp1d
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import directed_hausdorff
from typing import Dict, Tuple, Optional

try:
    from skimage.metrics import structural_similarity as _ssim
except Exception:
    _ssim = None


def compute_basic_metrics(actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Compute basic error metrics"""
    m = {}
    m['RMSE'] = np.sqrt(np.mean((actual - pred) ** 2))
    m['MAE'] = np.mean(np.abs(actual - pred))
    
    # MAPE with zero handling
    nz = actual != 0
    if np.any(nz):
        m['MAPE'] = np.mean(np.abs((actual[nz] - pred[nz]) / actual[nz])) * 100
    else:
        m['MAPE'] = np.nan
        
    m['Max_Error'] = np.max(np.abs(actual - pred))
    
    # R² Score
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    m['R2_Score'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return m


def compute_correlation_metrics(actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Compute correlation metrics"""
    m = {}
    
    # Pearson correlation
    m['Pearson_R'] = np.corrcoef(actual, pred)[0, 1]
    
    # Spearman rank correlation
    m['Spearman_R'] = stats.spearmanr(actual, pred)[0]
    
    # Cosine similarity
    dp = np.dot(actual, pred)
    na = np.linalg.norm(actual)
    npred = np.linalg.norm(pred)
    m['Cosine_Sim'] = dp / (na * npred) if na * npred != 0 else 0
    
    return m


def compute_structural_metrics(actual: np.ndarray, pred: np.ndarray, 
                              wl: np.ndarray) -> Dict[str, float]:
    """Compute structural similarity metrics"""
    m = {}
    
    # Normalize for SSIM
    a = (actual - actual.min()) / (actual.max() - actual.min() + 1e-10)
    p = (pred - pred.min()) / (pred.max() - pred.min() + 1e-10)
    
    # SSIM
    w = 7  # window size
    if _ssim is not None and len(a) >= w:
        a2 = np.tile(a, (w, 1))
        p2 = np.tile(p, (w, 1))
        m['SSIM'] = _ssim(a2, p2, win_size=w, data_range=1.0)
    else:
        # Fallback to correlation
        m['SSIM'] = np.corrcoef(a, p)[0, 1]
    
    # Multi-scale SSIM
    scales = [1, 2, 4]
    ms_vals = []
    for s in scales:
        if len(a) >= s * w:
            da = signal.resample(a, len(a) // s)
            dp = signal.resample(p, len(p) // s)
            if _ssim is not None and len(da) >= w:
                da2 = np.tile(da, (w, 1))
                dp2 = np.tile(dp, (w, 1))
                ms_vals.append(_ssim(da2, dp2, win_size=w, data_range=1.0))
    
    m['MS_SSIM'] = np.mean(ms_vals) if ms_vals else m['SSIM']
    
    # Spectral Angle Mapper (SAM)
    dp2 = np.dot(actual, pred)
    na = np.linalg.norm(actual)
    npd = np.linalg.norm(pred)
    ca = dp2 / (na * npd) if na * npd != 0 else 1
    m['SAM_radians'] = np.arccos(np.clip(ca, -1, 1))
    
    return m


def compute_distribution_metrics(actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Compute distribution distance metrics"""
    m = {}
    
    # Wasserstein distance
    m['Wasserstein'] = wasserstein_distance(actual, pred)
    
    # KL Divergence
    ap = actual - actual.min() + 1e-10
    ap = ap / ap.sum()
    pp = pred - pred.min() + 1e-10
    pp = pp / pp.sum()
    
    kl = stats.entropy(ap, pp)
    m['KL_Divergence'] = kl if not np.isinf(kl) else np.nan
    
    # Jensen-Shannon Distance
    mix = 0.5 * (ap + pp)
    jsd = 0.5 * stats.entropy(ap, mix) + 0.5 * stats.entropy(pp, mix)
    m['JS_Distance'] = np.sqrt(jsd)
    
    return m


def compute_spectral_metrics(actual: np.ndarray, pred: np.ndarray, 
                            wl: np.ndarray) -> Dict[str, float]:
    """Compute spectral-specific metrics"""
    m = {}
    
    # Peak wavelength error
    ai, pi = np.argmax(actual), np.argmax(pred)
    m['Peak_Lambda_Error_nm'] = abs(wl[ai] - wl[pi])
    m['Peak_Abs_Error'] = abs(actual[ai] - pred[pi])
    
    # FWHM difference
    def _fwhm(x):
        idx = np.argmax(x)
        half = x[idx] / 2
        idxs = np.where(x >= half)[0]
        return wl[idxs[-1]] - wl[idxs[0]] if len(idxs) else 0
    
    m['FWHM_Diff'] = abs(_fwhm(actual) - _fwhm(pred))
    
    # Area difference
    m['Area_Diff'] = abs(np.trapz(actual, wl) - np.trapz(pred, wl))
    
    return m


def compute_shape_metrics(actual: np.ndarray, pred: np.ndarray, 
                         wl: np.ndarray) -> Dict[str, float]:
    """Compute shape-based distance metrics"""
    m = {}
    
    # Dynamic Time Warping (DTW) distance
    n = len(actual)
    DTW = np.full((n, n), np.inf)
    DTW[0, 0] = abs(actual[0] - pred[0])
    
    for i in range(1, n):
        DTW[i, 0] = DTW[i-1, 0] + abs(actual[i] - pred[0])
    for j in range(1, n):
        DTW[0, j] = DTW[0, j-1] + abs(actual[0] - pred[j])
    
    for i in range(1, n):
        ai = actual[i]
        for j in range(1, n):
            c = abs(ai - pred[j])
            DTW[i, j] = c + min(DTW[i-1, j], DTW[i, j-1], DTW[i-1, j-1])
    
    m['DTW_Distance'] = DTW[n-1, n-1] / n
    
    # Fréchet distance (approximated by Hausdorff)
    curve_a = np.column_stack((wl, actual))
    curve_p = np.column_stack((wl, pred))
    dfw = directed_hausdorff(curve_a, curve_p)[0]
    dbw = directed_hausdorff(curve_p, curve_a)[0]
    m['Frechet_Dist'] = max(dfw, dbw)
    
    # Derivative MSE
    da = np.gradient(actual, wl)
    dp = np.gradient(pred, wl)
    m['Derivative_MSE'] = np.mean((da - dp) ** 2)
    
    return m


def compute_frequency_metrics(actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Compute frequency domain metrics"""
    fa, fp = np.fft.fft(actual), np.fft.fft(pred)
    ma, mp = np.abs(fa), np.abs(fp)
    
    out = {}
    
    # FFT correlation
    out['FFT_Correlation'] = np.corrcoef(ma, mp)[0, 1]
    
    # Power ratio
    psa, psp = np.abs(fa) ** 2, np.abs(fp) ** 2
    ta, tp = psa.sum(), psp.sum()
    out['Power_Ratio'] = tp / ta if ta != 0 else np.nan
    
    return out


def compute_all_metrics(actual: np.ndarray, pred: np.ndarray, 
                        wl: np.ndarray) -> Dict[str, float]:
    """
    Compute all 20+ metrics for spectral validation
    
    Args:
        actual: Ground truth spectrum
        pred: Predicted spectrum
        wl: Wavelength array
        
    Returns:
        Dictionary with all metrics
    """
    m = {}
    m.update(compute_basic_metrics(actual, pred))
    m.update(compute_correlation_metrics(actual, pred))
    m.update(compute_structural_metrics(actual, pred, wl))
    m.update(compute_distribution_metrics(actual, pred))
    m.update(compute_spectral_metrics(actual, pred, wl))
    m.update(compute_shape_metrics(actual, pred, wl))
    m.update(compute_frequency_metrics(actual, pred))
    
    return m


def interpolate_holdout(wl: np.ndarray, concs: list, abs_mat: np.ndarray, 
                       holdout_idx: int, conc_val: float) -> np.ndarray:
    """
    Basic interpolation for holdout concentration
    Matches A100 implementation - uses cubic extrapolation which fails at edges
    This demonstrates why geodesic approach is needed
    
    Args:
        wl: Wavelength array
        concs: All concentration values
        abs_mat: Absorbance matrix [wavelengths, concentrations]
        holdout_idx: Index to exclude
        conc_val: Target concentration value
        
    Returns:
        Interpolated spectrum
    """
    train_concs = [concs[i] for i in range(len(concs)) if i != holdout_idx]
    train_abs = np.column_stack([abs_mat[:, i] for i in range(len(concs)) if i != holdout_idx])
    
    y = np.zeros(len(wl))
    for i in range(len(wl)):
        # Use cubic if we have enough points, otherwise linear
        # This matches A100 implementation exactly
        if len(train_concs) >= 4:
            interp_func = interp1d(train_concs, train_abs[i, :], 
                                 kind='cubic', fill_value='extrapolate',
                                 bounds_error=False)
        else:
            interp_func = interp1d(train_concs, train_abs[i, :], 
                                 kind='linear', fill_value='extrapolate',
                                 bounds_error=False)
        
        y[i] = interp_func(conc_val)
    
    return y


# Metric categorization for analysis
HIGHER_BETTER = {
    'R2_Score', 'Pearson_R', 'Spearman_R', 'Cosine_Sim', 
    'SSIM', 'MS_SSIM', 'FFT_Correlation'
}

LOWER_BETTER = {
    'RMSE', 'MAE', 'MAPE', 'Max_Error', 'Wasserstein', 
    'KL_Divergence', 'JS_Distance', 'Peak_Lambda_Error_nm', 
    'Peak_Abs_Error', 'FWHM_Diff', 'Area_Diff', 'DTW_Distance', 
    'Frechet_Dist', 'Derivative_MSE', 'SAM_radians'
}


def compute_improvement(basic_val: float, improved_val: float, 
                       metric: str, eps: float = 1e-12) -> float:
    """
    Compute percentage improvement for a metric
    
    Args:
        basic_val: Basic method value
        improved_val: Improved method value
        metric: Metric name
        eps: Small value to avoid division by zero
        
    Returns:
        Percentage improvement
    """
    if metric in LOWER_BETTER:
        return 100 * (basic_val - improved_val) / (abs(basic_val) + eps)
    elif metric in HIGHER_BETTER:
        return 100 * (improved_val - basic_val) / (abs(basic_val) + eps)
    else:
        return np.nan