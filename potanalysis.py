# potanalysis.py
# Stand-alone Peaks-Over-Threshold utilities with NO external project deps.

import numpy as np
import pandas as pd
import scipy.stats as st
from typing import Tuple, Optional, Sequence

# ---------------------------
# Basic utilities
# ---------------------------

def ecdf(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Empirical CDF: returns sorted x and plotting positions y=(1..n)/(n+1)."""
    x = np.sort(np.asarray(data))
    n = x.size
    y = np.arange(1, n + 1) / (n + 1.0)
    return x, y

def record_length_years(df: pd.DataFrame) -> float:
    """
    Best-effort record length in years.
    Accepts either a 'time' (numeric, decimal years) or 'date' (datetime-like) column.
    Falls back to len(df)/365.25 if neither exists.
    """
    if 'time' in df.columns and np.issubdtype(df['time'].dtype, np.number):
        return float(df['time'].iloc[-1] - df['time'].iloc[0])
    if 'date' in df.columns:
        t = pd.to_datetime(df['date'])
        return (t.max() - t.min()).days / 365.25
    return len(df) / 365.25

# ---------------------------
# Declustering 
# ---------------------------

def _time_to_float_years(series: pd.Series) -> np.ndarray:
    """
    Convert a 'date' or 'time' column into monotone float years for gap checks.
    - If numeric -> assume decimal years and return as float
    - If datetime-like -> convert to ordinal days / 365.25
    """
    if np.issubdtype(series.dtype, np.number):
        return series.to_numpy(dtype=float)
    t = pd.to_datetime(series)
    base = t.min()
    return (t - base).dt.total_seconds().to_numpy() / (365.25 * 24 * 3600.0)

def clust(df: pd.DataFrame,
          u: float,
          time_cond: float,
          clust_max: bool = True) -> Tuple[pd.DataFrame, Optional[object]]:
    """
    Simple runs-declustering on exceedances over threshold u.

    Args
    ----
    df : DataFrame with at least 'obs' and either 'date' or 'time'
    u  : threshold (same units as 'obs')
    time_cond : minimum separation (in YEARS) between cluster peaks to be independent
                (e.g., 7/365 for ~one week). If your df['time'] is in decimal years,
                pass the gap directly in years too.
    clust_max : if True, return only cluster maxima; otherwise return rows per cluster

    Returns
    -------
    df_clusters : DataFrame of cluster maxima (includes original columns)
    ax : None (kept for compatibility; plotting removed for a clean core)
    """
    if 'obs' not in df.columns:
        raise ValueError("df must contain an 'obs' column.")

    # Build a unified time axis in float years
    if 'time' in df.columns:
        t = _time_to_float_years(df['time'])
    elif 'date' in df.columns:
        t = _time_to_float_years(df['date'])
    else:
        # assume equal spacing, 1/365.25 years per sample
        t = np.arange(len(df)) / 365.25

    # Keep exceedances only
    over = df['obs'].to_numpy() > u
    if not np.any(over):
        return df.iloc[[]].copy(), None

    dfu = df.loc[over].copy()
    tu = t[over]

    # Start first cluster at the first exceedance
    clusters_idx = []
    start = 0
    for i in range(1, len(dfu)):
        if (tu[i] - tu[i - 1]) > time_cond:
            clusters_idx.append((start, i - 1))
            start = i
    clusters_idx.append((start, len(dfu) - 1))

    # Take maxima per cluster
    rows = []
    for a, b in clusters_idx:
        seg = dfu.iloc[a:b + 1]
        idxmax = seg['obs'].idxmax()
        rows.append(idxmax)

    df_clusters = df.loc[rows].sort_index().copy()
    return df_clusters, None

# ---------------------------
# GPD 
# ---------------------------

def fit_gpd_mle(z: np.ndarray) -> Tuple[float, float]:
    """
    MLE for GPD on exceedances z (location fixed at 0).
    Returns (xi, sigma).
    """
    z = np.asarray(z, dtype=float)
    if np.any(z <= 0):
        raise ValueError("Exceedances must be strictly positive (z = x - u > 0).")
    xi, loc, sig = st.genpareto.fit(z, floc=0.0)
    return float(xi), float(sig)

def gpd_loglik(z: np.ndarray, xi: float, sig: float) -> float:
    """GPD log-likelihood for exceedances z at θ=(xi, sig), loc=0."""
    if sig <= 0:
        return -np.inf
    if np.any(1.0 + xi * z / sig <= 0.0):
        return -np.inf
    return float(np.sum(st.genpareto.logpdf(z, c=xi, loc=0.0, scale=sig)))

def mrl(z: Sequence[float],
        u_min: Optional[float] = None,
        u_max: Optional[float] = None,
        n: int = 50,
        conf: float = 0.95):
    """
    Mean Residual Life plot helper (no plotting; returns thresholds and CIs).
    """
    x = np.asarray(z, dtype=float)
    if u_min is None:
        u_min = np.min(x)
    if u_max is None:
        u_max = np.sort(x)[-4]  # avoid extreme tail instability
    thresholds = np.linspace(u_min, u_max, n)

    means, lo, hi = [], [], []
    zq = st.norm.ppf(0.5 + conf / 2.0)
    for u in thresholds:
        xu = x[x > u] - u
        if xu.size == 0:
            means += [np.nan]; lo += [np.nan]; hi += [np.nan]; continue
        m = float(xu.mean())
        s = float(m / np.sqrt(len(xu))) if len(xu) > 0 else np.nan
        means.append(m)
        lo.append(m - zq * s); hi.append(m + zq * s)
    return thresholds, np.asarray(means), np.asarray(lo), np.asarray(hi)

def threshold_choice(z: Sequence[float],
                     u_min: Optional[float] = None,
                     u_max: Optional[float] = None,
                     n: int = 30):
    """
    Threshold-choice helper: for a grid of u, fit GPD to (x>u) and return:
      - modified scale  σ_u* = σ - ξ u  (valid when fitting exceedances with loc=0)
      - shape ξ
    Stable (σ_u*, ξ) across u indicates a reasonable threshold range.
    """
    x = np.asarray(z, dtype=float)
    if u_min is None:
        u_min = float(np.min(x))
    if u_max is None:
        u_max = float(np.sort(x)[-4])
    thresholds = np.linspace(u_min, u_max, n)

    sigma_star, xi_list = [], []
    for u in thresholds:
        xu = x[x > u] - u
        if xu.size < 5:
            sigma_star.append(np.nan); xi_list.append(np.nan); continue
        xi, sig = fit_gpd_mle(xu)
        sigma_star.append(sig - xi * u)  # loc=0 exceedance parametrization
        xi_list.append(xi)
    return thresholds, np.asarray(sigma_star), np.asarray(xi_list)

# ---------------------------
# build exceedances from raw series
# ---------------------------

def make_exceedances(df: pd.DataFrame, u: float, time_cond: float) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Decluster df at threshold u and return (cluster maxima DataFrame, exceedances z=x-u>0).
    """
    df_cl, _ = clust(df, u=u, time_cond=time_cond, clust_max=True)
    z = (df_cl['obs'][df_cl['obs'] > u] - u).to_numpy(dtype=float)
    return df_cl, z

# ---------------------------
# Minimal demo
# ---------------------------

if __name__ == "__main__":
    # Minimal example; safe to remove. Adjust CSV path and parameters if you want to test quickly.
    import sys
    import matplotlib.pyplot as plt

    if len(sys.argv) < 2:
        print("Usage: python potanalysis.py <csv_path>  [u=12.0]  [gap_days=7]")
        sys.exit(0)

    CSV = sys.argv[1]
    u = float(sys.argv[2]) if len(sys.argv) >= 3 else 12.0
    gap_days = float(sys.argv[3]) if len(sys.argv) >= 4 else 7.0
    time_cond = gap_days / 365.25

    df = pd.read_csv(CSV).dropna()
    df_cl, z = make_exceedances(df, u=u, time_cond=time_cond)
    print(f"# clusters above u: {len(z)}   (u={u})")

    xi, sig = fit_gpd_mle(z)
    print(f"MLE GPD: xi={xi:.3f}, sigma={sig:.3f}")

    T, m, lo, hi = mrl(df['obs'].values, u_min=u, u_max=np.percentile(df['obs'], 95), n=40)
    Tu, sigstar, xis = threshold_choice(df['obs'].values, u_min=u, u_max=np.percentile(df['obs'], 95), n=25)

    # quick plots
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    ax[0].plot(T, m, label='MRL'); ax[0].fill_between(T, lo, hi, alpha=0.2); ax[0].axvline(u, ls='--', c='k'); ax[0].set_title("MRL")
    ax[1].plot(Tu, sigstar, label="σ*"); ax[1].axvline(u, ls='--', c='k'); ax[1].set_title("Threshold choice (σ*)")
    ax[2].plot(Tu, xis, label="xi"); ax[2].axvline(u, ls='--', c='k'); ax[2].set_title("Threshold choice (ξ)")
    for a in ax: a.legend(); a.set_xlabel("threshold u")
    plt.tight_layout(); plt.show()
