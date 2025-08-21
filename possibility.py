# possibility.py
import numpy as np
from scipy.optimize import minimize
import scipy.stats as st

# ---------- Poisson rate (λ0) from N events in T years ----------
def lambda_possibility_from_counts(N, T, lam_grid=None):
    lam_hat = N / T
    if lam_grid is None:
        lo = max(1e-8, 0.1*lam_hat) if lam_hat>0 else 1e-6
        hi = 5*lam_hat + 1e-6 if lam_hat>0 else 5.0
        lam_grid = np.linspace(lo, hi, 2000)
    # log-likelihood up to additive constant
    ell = N*np.log(np.maximum(lam_grid*T, 1e-300)) - lam_grid*T
    ell -= np.max(ell)
    pi = np.exp(ell)
    return lam_hat, lam_grid, pi

def alpha_interval_1d(grid, pi, alpha):
    m = (pi >= alpha)
    if not np.any(m): return None
    return float(grid[m].min()), float(grid[m].max())

# ---------- GPD(ξ, σ) possibility from exceedances ----------
def gpd_loglik(x_excess, xi, sigma):
    if sigma <= 0: return -np.inf
    if np.any(1 + xi * x_excess / sigma <= 0): return -np.inf
    return np.sum(st.genpareto.logpdf(x_excess, c=xi, loc=0, scale=sigma))

def gpd_possibility_grid(x_excess, xi_grid, sig_grid):
    XI, SIG = np.meshgrid(xi_grid, sig_grid, indexing='ij')
    L = np.full_like(XI, -np.inf, dtype=float)
    for i in range(XI.shape[0]):
        for j in range(XI.shape[1]):
            L[i, j] = gpd_loglik(x_excess, XI[i, j], SIG[i, j])
    L -= np.max(L)
    PI = np.exp(L)  # normalize to sup = 1
    return XI, SIG, PI

def gpd_quantile_alpha_band(Tr, u, alpha, xi_grid, sig_grid, PI, lam_interval):
    """
    Return α-cut band for discharge Q_Tr (per-year return period Tr) using:
    Q_Tr = u + GPD.ppf(1 - 1/(λ0*Tr), ξ, σ)
    """
    # λ0 α-cut interval (min/max); if None, treat as single value
    lam_lo, lam_hi = lam_interval if lam_interval is not None else (None, None)

    # candidate p's over α-cut of λ0
    if lam_lo is None:
        lam_candidates = []
    else:
        lam_candidates = [lam_lo, lam_hi]

    # collect all q from (ξ,σ) with π≥α and λ0 in α-cut
    mask = (PI >= alpha)
    xi_vals = xi_grid[mask]
    sig_vals = sig_grid[mask]
    qs = []

    # if λ0 uncertain, consider both ends (worst/best cases)
    lam_ends = lam_candidates if lam_candidates else [None]
    for lam in lam_ends:
        if lam is None:
            # fallback: treat Tr as if p=1/Tr (only if you prefer to ignore λ0 here)
            p_ex = 1.0/Tr
        else:
            p_ex = 1.0/(lam*Tr)
        p_ex = np.clip(p_ex, 1e-12, 1-1e-12)
        q_ex = st.genpareto.ppf(1 - p_ex, c=xi_vals, loc=0, scale=sig_vals)
        qs.append(u + q_ex)

    q_all = np.concatenate(qs) if len(qs)>1 else qs[0]
    return float(np.min(q_all)), float(np.max(q_all))

# ---------- GLUE → possibility over β=(r_ch, r_fp) ----------
def glue_scores_to_possibility(F_scores):
    # scale to [0,1] and renormalize to sup=1
    F = np.asarray(F_scores, dtype=float)
    if np.allclose(F.max(), F.min()):
        pi = np.ones_like(F)
    else:
        pi = (F - F.min())/(F.max() - F.min())
    if pi.max() > 0:
        pi /= pi.max()
    return pi

def alpha_subset_indices(pi, alpha):
    return np.where(pi >= alpha)[0]
