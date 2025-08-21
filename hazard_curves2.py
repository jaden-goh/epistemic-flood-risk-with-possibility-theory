# hazard_curves.py
# -----------------------------------------------------------------------------
# Possibilistic hazard pipeline:
#   - Decluster series, compute exceedances above u
#   - Build possibility distributions π(λ0), π(ξ,σ), π(β) (GLUE)
#   - For each α and return period Tr, compute discharge α-band [Qmin,Qmax]
#   - Propagate (Qmin/Qmax) with β in α-cut through Lisflood -> depth α-bands
# Outputs:
#   - Qbands: dict[alpha] -> array nTr x 2 (Qmin,Qmax)
#   - DepthBands: dict[alpha] -> array nTr x 2 x Ny x Nx (min/max depth maps)
# -----------------------------------------------------------------------------

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st

from typing import Dict, Tuple, Optional

# --- repo-local imports (keep from your codebase)
from potanalysis import clust  # declustering
from flood_simulator import Lisflood  # Lisflood('array', Q, Hfix, [r_ch, r_fp])

# =========================
# Configuration (edit here)
# =========================
DATA_CSV = "data/ardieres.csv"   # input discharge series
GLUE_RESULTS_CSV = "results.csv" # output of lisflood_grid_simulation.py
OUTDIR = "outputs/poss_hazard"   # where to store bands/figures

# Declustering / POT
U_THRESHOLD = 12.0               # threshold u for exceedances (same units as obs)
TIME_COND = 7/365.0              # min inter-event gap in "years" if df['time'] is decimal years; adjust if needed

# GPD grids (tune to your site)
XI_GRID = np.linspace(-0.30, 0.60, 120)   # GPD shape (ξ)
SIG_GRID = np.linspace(  5.00, 40.00, 150) # GPD scale (σ), with loc fixed at 0

# α-levels and Return Periods
ALPHAS = [0.90, 0.75, 0.50]
TRS = np.array([1, 2, 5, 10, 25, 50, 100, 250, 500, 750, 1000], dtype=float)

# Lisflood propagation
HFIX = 68.43                       # your stage/BC head if needed by Lisflood
BETA_SAMPLE_PER_ALPHA = 24         # number of β=(rch,rfp) samples per α (trade-off speed/coverage)
CROP_BOUNDS = None                 # [y0, y1, x0, x1] or None to keep full grid

# Reproducibility
RNG_SEED = 12345

# =============================================================================
# Helpers: record length, possibility constructions, α-cut utilities
# =============================================================================

def record_length_years(df: pd.DataFrame) -> float:
    """Best-effort record length in years using 'time' (decimal years) or 'date' (datetime)."""
    if 'time' in df.columns and np.issubdtype(df['time'].dtype, np.number):
        # assume decimal years
        return float(df['time'].iloc[-1] - df['time'].iloc[0])
    if 'date' in df.columns:
        t = pd.to_datetime(df['date'])
        return (t.max() - t.min()).days / 365.25
    # fallback: assume daily index
    return len(df) / 365.25

def lambda_possibility_from_counts(N: int, T_years: float,
                                   lam_grid: Optional[np.ndarray]=None
                                   ) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Possibility distribution for Poisson rate λ0 given count N over exposure T:
      π(λ) = exp( ℓ(λ) - max ℓ ), with ℓ(λ) = N log(λT) - λT (up to const).
    """
    lam_hat = N / T_years if T_years > 0 else np.nan
    if lam_grid is None:
        if lam_hat > 0:
            lo = max(1e-8, 0.1 * lam_hat)
            hi = 5.0 * lam_hat + 1e-6
        else:
            lo, hi = 1e-6, 5.0
        lam_grid = np.linspace(lo, hi, 2000)
    ell = N * np.log(np.maximum(lam_grid * T_years, 1e-300)) - lam_grid * T_years
    ell -= np.max(ell)
    pi = np.exp(ell)
    return lam_hat, lam_grid, pi

def alpha_interval_1d(grid: np.ndarray, pi: np.ndarray, alpha: float
                      ) -> Optional[Tuple[float, float]]:
    m = (pi >= alpha)
    if not np.any(m):
        return None
    return float(grid[m].min()), float(grid[m].max())

def gpd_loglik(x_excess: np.ndarray, xi: float, sigma: float) -> float:
    if sigma <= 0:
        return -np.inf
    # validity domain: 1 + ξ x / σ > 0 for all x
    if np.any(1.0 + xi * x_excess / sigma <= 0.0):
        return -np.inf
    return float(np.sum(st.genpareto.logpdf(x_excess, c=xi, loc=0.0, scale=sigma)))

def gpd_possibility_grid(x_excess: np.ndarray,
                         xi_grid: np.ndarray, sig_grid: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    XI, SIG = np.meshgrid(xi_grid, sig_grid, indexing='ij')
    L = np.full_like(XI, -np.inf, dtype=float)
    for i in range(XI.shape[0]):
        for j in range(XI.shape[1]):
            L[i, j] = gpd_loglik(x_excess, XI[i, j], SIG[i, j])
    L -= np.max(L)
    PI = np.exp(L)  # normalize to sup = 1
    return XI, SIG, PI

def gpd_quantile_alpha_band(Tr: float, u: float, alpha: float,
                            XI: np.ndarray, SIG: np.ndarray, PI: np.ndarray,
                            lam_interval: Optional[Tuple[float,float]]
                            ) -> Tuple[float, float]:
    """
    α-cut band for Q_Tr using:
      Q_Tr = u + GPD^{-1}(1 - 1/(λ0*Tr), ξ, σ).
    We envelope over (ξ,σ) with π>=α and over λ0 in its α-cut interval.
    """
    mask = (PI >= alpha)
    if not np.any(mask):
        return (np.nan, np.nan)
    xi_vals = XI[mask]
    sig_vals = SIG[mask]

    # handle λ0 α-interval (if missing, fall back to p=1/Tr)
    lam_ends = []
    if lam_interval is not None and lam_interval[0] is not None:
        lam_ends = [lam_interval[0], lam_interval[1]]
    else:
        lam_ends = [None]

    qs = []
    for lam in lam_ends:
        p_ex = (1.0 / Tr) if lam is None else (1.0 / (lam * Tr))
        p_ex = float(np.clip(p_ex, 1e-12, 1.0 - 1e-12))
        # GPD exceedance quantile
        q_ex = st.genpareto.ppf(1.0 - p_ex, c=xi_vals, loc=0.0, scale=sig_vals)
        qs.append(u + q_ex)

    q_all = np.concatenate(qs) if len(qs) > 1 else qs[0]
    return float(np.nanmin(q_all)), float(np.nanmax(q_all))

def glue_scores_to_possibility(df_glue: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Turn GLUE results into π(β). Accepts either:
      - df with 'F' (score) & rch/rfp columns, or
      - df with 'A','B','C' & rch/rfp (we compute F=(A-B)/(A+B+C)).
    Returns:
      beta_pairs: (N,2) array of [rch, rfp]
      pi_beta:   (N,)   membership scaled to [0,1] with sup=1
    """
    if 'F' in df_glue.columns:
        F = df_glue['F'].values.astype(float)
    elif all(c in df_glue.columns for c in ['A','B','C']):
        A = df_glue['A'].values.astype(float)
        B = df_glue['B'].values.astype(float)
        C = df_glue['C'].values.astype(float)
        denom = A + B + C
        with np.errstate(divide='ignore', invalid='ignore'):
            F = (A - B) / np.where(denom == 0, np.nan, denom)
        F = np.nan_to_num(F, nan=0.0)
    else:
        raise ValueError("GLUE CSV must contain 'F' or the trio 'A','B','C'.")

    # scale to [0,1], renormalize sup=1
    if np.isclose(F.max(), F.min()):
        pi = np.ones_like(F, dtype=float)
    else:
        pi = (F - F.min()) / (F.max() - F.min())
    if pi.max() > 0:
        pi /= pi.max()

    # beta pairs
    # Try flexible column names
    rch_col = 'rch' if 'rch' in df_glue.columns else 'r_ch'
    rfp_col = 'rfp' if 'rfp' in df_glue.columns else 'r_fp'
    if not (rch_col in df_glue.columns and rfp_col in df_glue.columns):
        raise ValueError("GLUE CSV must include columns for channel/floodplain roughness (rch/r_ch and rfp/r_fp).")
    beta_pairs = df_glue[[rch_col, rfp_col]].values.astype(float)
    return beta_pairs, pi

def alpha_subset_indices(pi: np.ndarray, alpha: float) -> np.ndarray:
    return np.where(pi >= alpha)[0]

def maybe_crop(arr: np.ndarray, bounds: Optional[Tuple[int,int,int,int]]):
    if bounds is None:
        return arr
    y0, y1, x0, x1 = bounds
    return arr[y0:y1+1, x0:x1+1]

# =============================================================================
# Main routine
# =============================================================================

def run_possibilistic_hazard():
    rng = np.random.default_rng(RNG_SEED)
    os.makedirs(OUTDIR, exist_ok=True)

    # -----------------------
    # 1) Load and decluster
    # -----------------------
    df = pd.read_csv(DATA_CSV).dropna()
    # Declustering on raw series, then select exceedances above u
    df_cl, _ = clust(df, u=U_THRESHOLD, time_cond=TIME_COND, clust_max=True, plot=False)
    qu = (df_cl['obs'][df_cl['obs'] > U_THRESHOLD] - U_THRESHOLD).values
    if qu.size == 0:
        raise RuntimeError("No exceedances above threshold u. Adjust U_THRESHOLD.")

    # Exposure T and event count N for λ0
    T_years = record_length_years(df)
    N_events = int((df_cl['obs'] > U_THRESHOLD).sum())

    # -----------------------
    # 2) π(λ0) from counts
    # -----------------------
    lam_hat, lam_grid, pi_lam = lambda_possibility_from_counts(N_events, T_years)
    lam_alpha: Dict[float, Optional[Tuple[float,float]]] = {
        a: alpha_interval_1d(lam_grid, pi_lam, a) for a in ALPHAS
    }

    # Quick plot (optional)
    plt.figure(figsize=(6,3))
    plt.plot(lam_grid, pi_lam, label=r'$\pi(\lambda_0)$')
    for a in ALPHAS:
        iv = lam_alpha[a]
        if iv is not None:
            plt.axvspan(iv[0], iv[1], alpha=0.1, label=f'α={a:.2f}')
    plt.axvline(lam_hat, color='r', linestyle='--', label=r'$\hat\lambda_0$')
    plt.xlabel('λ₀ (events/year)'); plt.ylabel('Possibility')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "lambda0_possibility.png"), dpi=160)
    plt.close()

    # -----------------------
    # 3) π(ξ,σ) on a grid
    # -----------------------
    XI, SIG, PI = gpd_possibility_grid(qu, XI_GRID, SIG_GRID)

    # Save a contour snapshot for sanity
    plt.figure(figsize=(6,5))
    cs = plt.contourf(SIG, XI, PI, levels=np.linspace(0,1,11), cmap='viridis')
    plt.colorbar(cs, label='π(ξ,σ)')
    plt.xlabel('σ'); plt.ylabel('ξ'); plt.title('GPD possibility (loc=0)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "gpd_possibility.png"), dpi=160)
    plt.close()

    # -----------------------
    # 4) GLUE -> π(β)
    # -----------------------
    if not os.path.exists(GLUE_RESULTS_CSV):
        raise FileNotFoundError(f"GLUE results CSV not found: {GLUE_RESULTS_CSV}")
    df_glue = pd.read_csv(GLUE_RESULTS_CSV).dropna()
    beta_pairs, pi_beta = glue_scores_to_possibility(df_glue)

    # -----------------------
    # 5) Q-bands per α, per Tr
    # -----------------------
    Qbands: Dict[float, np.ndarray] = {a: np.zeros((len(TRS), 2), dtype=float) for a in ALPHAS}
    for a in ALPHAS:
        lam_iv = lam_alpha[a]
        for k, Tr in enumerate(TRS):
            q_lo, q_hi = gpd_quantile_alpha_band(Tr, U_THRESHOLD, a, XI, SIG, PI, lam_iv)
            Qbands[a][k, :] = (q_lo, q_hi)

    # Save Qbands
    for a in ALPHAS:
        dfQ = pd.DataFrame({
            "Tr": TRS,
            "Qmin": Qbands[a][:,0],
            "Qmax": Qbands[a][:,1],
        })
        dfQ.to_csv(os.path.join(OUTDIR, f"Qbands_alpha_{a:.2f}.csv"), index=False)

    # Quick plot
    plt.figure(figsize=(7,4))
    for a in ALPHAS:
        plt.fill_between(TRS, Qbands[a][:,0], Qbands[a][:,1], alpha=0.15, label=f'α={a:.2f}')
    plt.xscale('log')
    plt.xlabel('Return period Tr [years]')
    plt.ylabel('Discharge Q [units of obs]')
    plt.title('Possibilistic discharge bands by return period')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "Qbands.png"), dpi=160)
    plt.close()

    # -----------------------
    # 6) Propagate through Lisflood to get depth α-bands
    #     Strategy: for each α, for each Tr:
    #       run Lisflood at Qmin and Qmax, across a subset of β in α-cut,
    #       take pixelwise min/max envelope -> depth band
    # -----------------------
    DepthBands: Dict[float, np.ndarray] = {
        a: None for a in ALPHAS
    }

    for a in ALPHAS:
        # β α-cut subset
        idx = alpha_subset_indices(pi_beta, a)
        if idx.size == 0:
            print(f"[warn] No GLUE β in α-cut for α={a:.2f}; skipping.")
            continue
        if idx.size > BETA_SAMPLE_PER_ALPHA:
            idx = rng.choice(idx, size=BETA_SAMPLE_PER_ALPHA, replace=False)
        beta_sub = beta_pairs[idx, :]

        # We need grid shape; do one dry run with some Q,β to get array dims
        probe_Q = float(np.nanmedian(Qbands[a])) if np.isfinite(Qbands[a]).any() else float(np.nanmedian(TRS))
        probe = Lisflood('array', probe_Q, HFIX, [beta_sub[0,0], beta_sub[0,1]])
        probe = maybe_crop(probe, CROP_BOUNDS)
        Ny, Nx = probe.shape

        depth_alpha = np.zeros((len(TRS), 2, Ny, Nx), dtype=float)

        for k, Tr in enumerate(TRS):
            qmin, qmax = Qbands[a][k, :]
            if not np.isfinite(qmin) or not np.isfinite(qmax):
                depth_alpha[k, 0] = np.nan
                depth_alpha[k, 1] = np.nan
                continue

            # Collect depth rasters across β for Qmin and Qmax
            Dmins = []
            Dmaxs = []
            for (rch, rfp) in beta_sub:
                Smin = Lisflood('array', float(qmin), HFIX, [float(rch), float(rfp)])
                Smax = Lisflood('array', float(qmax), HFIX, [float(rch), float(rfp)])
                Smin = maybe_crop(Smin, CROP_BOUNDS)
                Smax = maybe_crop(Smax, CROP_BOUNDS)
                Dmins.append(Smin)
                Dmaxs.append(Smax)

            Dlo = np.min(np.stack(Dmins, axis=0), axis=0)
            Dhi = np.max(np.stack(Dmaxs, axis=0), axis=0)
            depth_alpha[k, 0] = Dlo
            depth_alpha[k, 1] = Dhi

        DepthBands[a] = depth_alpha
        # Persist to .npy (compact and fast)
        np.save(os.path.join(OUTDIR, f"DepthBands_alpha_{a:.2f}.npy"), depth_alpha)

    # -----------------------
    # 7) Manifest / metadata
    # -----------------------
    meta = {
        "data_csv": DATA_CSV,
        "glue_csv": GLUE_RESULTS_CSV,
        "u_threshold": U_THRESHOLD,
        "time_cond": TIME_COND,
        "alphas": ALPHAS,
        "Trs": TRS.tolist(),
        "xi_grid": [float(XI_GRID[0]), float(XI_GRID[-1]), int(XI_GRID.size)],
        "sig_grid": [float(SIG_GRID[0]), float(SIG_GRID[-1]), int(SIG_GRID.size)],
        "hfix": HFIX,
        "beta_sample_per_alpha": BETA_SAMPLE_PER_ALPHA,
        "crop_bounds": CROP_BOUNDS,
        "rng_seed": RNG_SEED,
        "N_events": N_events,
        "T_years": T_years,
        "lambda_hat": lam_hat
    }
    with open(os.path.join(OUTDIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] Possibilistic hazard complete. Outputs in: {OUTDIR}")

# =============================================================================
# Entrypoint
# =============================================================================
if __name__ == "__main__":
    run_possibilistic_hazard()
