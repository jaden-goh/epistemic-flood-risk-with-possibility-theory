
# possibilistic prediction with GPD (and optional Poisson rate).

import numpy as np
import scipy.stats as st


# ------------------------------
# 1) Possibilistic posterior π(ξ,σ | D) from normalized GPD likelihood
# ------------------------------
def gpd_posterior_possibility(z_excess, xi_grid, sig_grid):
    """
    Build the posterior possibility π(ξ,σ | D) by sup-normalizing the GPD likelihood.

    Args:
      z_excess (1D array): exceedances z_i = x_i - u > 0
      xi_grid  (1D array): grid of shape parameters ξ
      sig_grid (1D array): grid of scale parameters σ (>0)

    Returns:
      XI, SIG (2D meshgrids), PI (2D) where PI[i,j] = π(ξ_i, σ_j | D) in [0,1], sup=1
    """
    XI, SIG = np.meshgrid(xi_grid, sig_grid, indexing="ij")
    L = np.full_like(XI, -np.inf, dtype=float)

    # Evaluate log-likelihood only where GPD is valid: 1 + ξ z / σ > 0 for all z
    for i in range(XI.shape[0]):
        for j in range(XI.shape[1]):
            xi = XI[i, j]
            sig = SIG[i, j]
            if sig <= 0:
                continue
            if np.any(1.0 + xi * z_excess / sig <= 0.0):
                continue
            L[i, j] = np.sum(st.genpareto.logpdf(z_excess, c=xi, loc=0.0, scale=sig))

    # sup-normalize ⇒ π(θ|D) = exp(ℓ(θ) - max ℓ) ∈ [0,1], with sup = 1
    L -= np.max(L)
    PI = np.exp(L)
    return XI, SIG, PI


# ------------------------------
# 2) Event probability under parameters: Pθ(B)
# ------------------------------
def event_prob_theta(xi, sig, u, B):
    """
    Closed-form event probability under θ=(ξ,σ) using the GPD CDF.

    B can be:
      ("tail", y)        : P( X >= y | θ ) with y >= u
      ("interval", a, b) : P( a <= X <= b | θ ), where a >= u
    """
    if sig <= 0:
        return 0.0

    if B[0] == "tail":
        y = float(B[1])
        if y <= u:
            return 1.0
        p = 1.0 - st.genpareto.cdf(y - u, c=xi, loc=0.0, scale=sig)
        return float(np.clip(p, 0.0, 1.0))

    elif B[0] == "interval":
        a, b = float(B[1]), float(B[2])
        if b <= u:
            return 0.0
        a = max(a, u)
        Fa = st.genpareto.cdf(a - u, c=xi, loc=0.0, scale=sig)
        Fb = st.genpareto.cdf(b - u, c=xi, loc=0.0, scale=sig)
        p = Fb - Fa
        return float(np.clip(p, 0.0, 1.0))

    else:
        raise ValueError("Unknown event set B")


# ------------------------------
# 3) Upper probability (outer measure) and necessity (lower probability)
#    — magnitudes only (no event rate)
# ------------------------------
def outer_and_necessity(z_excess, u, xi_grid, sig_grid, B, tnorm="product"):
    """
    Compute:
      P̄(B|D)   = sup_θ T( π(θ|D), Pθ(B) )
      P_(B|D)   = 1 - sup_θ T( π(θ|D), 1 - Pθ(B) )   (necessity dual)

    where T is the t-norm ("product" or "min").

    Returns:
      Pbar, Punder
    """
    XI, SIG, PI = gpd_posterior_possibility(z_excess, xi_grid, sig_grid)

    # Compute Pθ(B) on the same grid (skip impossible θ where π=0)
    P = np.zeros_like(PI)
    for i in range(XI.shape[0]):
        for j in range(SIG.shape[1]):
            if PI[i, j] == 0.0:
                continue
            P[i, j] = event_prob_theta(XI[i, j], SIG[i, j], u, B)

    if tnorm == "product":
        comb = PI * P
        comb_comp = PI * (1.0 - P)
    elif tnorm == "min":
        comb = np.minimum(PI, P)
        comb_comp = np.minimum(PI, 1.0 - P)
    else:
        raise ValueError("tnorm must be 'product' or 'min'")

    Pbar = float(np.max(comb))           # upper probability (outer measure)
    Pbar_comp = float(np.max(comb_comp)) # upper probability of complement
    Punder = float(1.0 - Pbar_comp)      # necessity (lower probability)
    return Pbar, Punder


# ------------------------------
# 4) Optional: include event rate λ0 for one-year exceedance
# ------------------------------
def lambda_possibility_from_counts(N, T_years, lam_grid=None):
    """
    π(λ0|D) from Poisson likelihood sup-normalization.
    """
    lam_hat = N / T_years
    if lam_grid is None:
        lo = max(1e-8, 0.1 * lam_hat) if lam_hat > 0 else 1e-6
        hi = 5.0 * lam_hat + 1e-6 if lam_hat > 0 else 5.0
        lam_grid = np.linspace(lo, hi, 2000)

    ell = N * np.log(np.maximum(lam_grid * T_years, 1e-300)) - lam_grid * T_years
    ell -= np.max(ell)
    pi = np.exp(ell)  # sup=1
    return lam_grid, pi


def outer_and_necessity_per_year(z_excess, u, xi_grid, sig_grid,
                                 N, T_years, B, tnorm="product"):
    """
    Same as outer_and_necessity but for one-year exceedance probability:
      P^(1yr)_{λ0,θ}(B) = 1 - exp( -λ0 * Pθ(B) )
    Combine π(λ0|D) and π(θ|D) with the chosen t-norm.
    """
    XI, SIG, PI_theta = gpd_posterior_possibility(z_excess, xi_grid, sig_grid)
    lam_grid, PI_lam = lambda_possibility_from_counts(N, T_years)

    Pbar = 0.0
    Pbar_comp = 0.0

    for i in range(XI.shape[0]):
        for j in range(SIG.shape[1]):
            if PI_theta[i, j] == 0.0:
                continue
            pB = event_prob_theta(XI[i, j], SIG[i, j], u, B)  # probability per exceedance
            Py = 1.0 - np.exp(-lam_grid * pB)                 # 1-year exceedance

            if tnorm == "product":
                comb = PI_theta[i, j] * (PI_lam * Py)
                comb_comp = PI_theta[i, j] * (PI_lam * (1.0 - Py))
            elif tnorm == "min":
                comb = np.minimum(PI_theta[i, j], np.minimum(PI_lam, Py))
                comb_comp = np.minimum(PI_theta[i, j], np.minimum(PI_lam, 1.0 - Py))
            else:
                raise ValueError("tnorm must be 'product' or 'min'")

            Pbar = max(Pbar, float(np.max(comb)))
            Pbar_comp = max(Pbar_comp, float(np.max(comb_comp)))

    Punder = float(1.0 - Pbar_comp)
    return Pbar, Punder
