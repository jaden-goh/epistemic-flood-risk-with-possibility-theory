import numpy as np
import pandas as pd
from potanalysis import clust 
from predict import outer_and_necessity, outer_and_necessity_per_year

# ------------------------------
# User/config inputs
# ------------------------------
DATA_CSV  = "ardieres.csv"
u         = 12.0          # POT threshold (same units as 'obs')
time_cond = 7/365.0       # declustering gap in years (≈ one week)
xi_grid   = np.linspace(-0.30, 0.60, 121)   # GPD ξ grid
sig_grid  = np.linspace(  5.00, 40.00, 141) # GPD σ grid
B         = ("tail", 25.0)                  # event set: exceed y = 25.0
tnorm     = "product"                       # or "min"

# ------------------------------
# Load and decluster
# ------------------------------
df = pd.read_csv(DATA_CSV).dropna()
df_cl, _ = clust(df, u=u, time_cond=time_cond, clust_max=True)

# Exceedances
z = (df_cl['obs'][df_cl['obs'] > u] - u).values
if z.size == 0:
    raise RuntimeError("No exceedances above u; choose a lower u or check data.")

# ------------------------------
# Magnitude
# ------------------------------
Pbar, Punder = outer_and_necessity(z, u, xi_grid, sig_grid, B, tnorm=tnorm)
print(f"Upper probability  P̄(X ∈ B | D) = {Pbar:.6f}")
print(f"Necessity (lower)  P_(X ∈ B | D) = {Punder:.6f}")

# ------------------------------
# one-year exceedance (event rate)
# ------------------------------
# Event count N and exposure T_years
N = int((df_cl['obs'] > u).sum())
if 'time' in df.columns and np.issubdtype(df['time'].dtype, np.number):
    T_years = float(df['time'].iloc[-1] - df['time'].iloc[0])
else:
    if 'date' in df.columns:
        t = pd.to_datetime(df['date'])
        T_years = (t.max() - t.min()).days / 365.25
    else:
        T_years = len(df) / 365.25

Pbar_y, Punder_y = outer_and_necessity_per_year(
    z, u, xi_grid, sig_grid, N, T_years, B, tnorm=tnorm
)
print(f"[Per-year] Upper = {Pbar_y:.6f}")
print(f"[Per-year] Lower = {Punder_y:.6f}")
