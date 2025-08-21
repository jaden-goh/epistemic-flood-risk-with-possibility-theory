import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats as st
import pandas as pd
import seaborn as sns
import osgeo.gdal as gdal
import pickle
import os
from potanalysis import clust, autocorr, MRL, TC, diagnostic_plots, ecdf
from potanalysis import bayesian_diagnostic_plots
from mh_posterior import mh_paths_sampler, paths_diagnostics
from flood_simulator import Lisflood, read_flood_results

"""
SIMULATION OF TIME SERIES OF INUNDATIONS

This script aims to compute a simulation of T years of extreme discharge events and the
associated inundations. The model's parameters are drawn from their respective posterior
distributions. It does this in 3 main steps:

1. Discharge frequency model: Bayesian POT approach from daily discharge historical data.
2. Flood simulator calibration: roughness parameters and their posterior distribution are 
    calibrated using GLUE.
3. Time series simulation: T years of events are simulated. For each event, a discharge
    is sampled from its predictive posterior distribution, and for each discharge an
    inundation is simulated using a posterior sample of roughness.

"""


## =============================================================================
# 1. DISCHARGE FREQUENCY MODEL

# Read discharge data
filename = 'data/Buscot-daily_discharge.csv'
df = pd.read_csv( filename, usecols=[0,1], skiprows=19, parse_dates=[0] )

# Add 'time' and 'obs' index columns
df['obs'] = df['discharge']
df['time'] = np.arange( 0, len(df), 1 )

# Plot time series
fig, ax = plt.subplots()
ax.plot( df['date'], df['discharge'], linewidth=0.5 )

# Autocorrelation of the series
acf, ax = autocorr( df['discharge'], plot=True );

# Fit GPD to exceedance points with selected threshold u
u = 12
df_cl, ax_clust = clust( df, u=u, time_cond=7, clust_max=True, plot=True)
qu = df_cl['obs'][ df_cl['obs']>u ] - u
acf, ax = autocorr( qu, plot=True );

# Mean recurrence interval
ti = np.diff( df_cl.time )
n = len(qu) # Events occurred
T = len(df)/365 # Total time in years
m_mean = (n+1/2)/T
m_dist = st.gamma( n+1/2, loc=0, scale=1/(T+1/2) )
m_pred = st.lomax( n+1/2, loc=0, scale=T+1/2 )

# Check exponential distribution of interarrival times
distr = st.expon
ti_params = distr.fit( ti )
diagnostic_plots( ti, distr, ti_params );

# MLE estimates
t_mle = st.genpareto.fit( qu, floc=0 )
# Priors (Jeffrey's prior from)
logPrior = lambda t: -np.log(t[1]) - np.log(1+t[0]) - 1/2*np.log(1+2*t[0])
t1_prior = st.norm( loc=3*t_mle[0], scale=np.abs(t_mle[0]*20) )
t2_prior = st.norm( loc=t_mle[2], scale=np.abs(t_mle[2]*2) )
logpriors = [t1_prior, t2_prior]
# Log-likelihood
logLikelihood = lambda t: sum( st.genpareto.logpdf(qu, t[0], loc=0, scale=t[1]) )
# Posterior
target_logpdf = lambda t: logLikelihood( t ) + logPrior( t )
# MCMC
Npaths = 4; Nsim = 15000; burnin = int(Nsim/2+1)
x0 = np.random.uniform( [-0.2,14], [0.2,20], [Npaths+1,2] )
sigmas = np.array([t_mle[0]*1.5, t_mle[2]*1.5])
cov = sigmas**2*np.eye(len(sigmas))
Xbin, Xstack, acceptance, _ = mh_paths_sampler(target_logpdf, Npaths, 
                                               Nsim, x0, cov,
                                               burnin=burnin, thin=1,
                                               sampler='adaptive', tune=400,
                                               tune_intvl=4000,
                                               proposal=None)
# MCMC convergence diagnostics
R, var_j, rhot, neff = paths_diagnostics( Xbin, True, *logpriors )
# Fit diagnostics
qFi, qTri, fig = bayesian_diagnostic_plots( qu, st.genpareto, Xstack, conf=0.9 )

## ====================================================================================
# 2. FLOOD SIMULATOR

# Training Data
bounds = [0,48,0,76]
src = gdal.Open( 'data/BuscotFlood92.tiff' )
z = src.GetRasterBand(1).ReadAsArray()
src = None
z = z[bounds[0]:bounds[1],bounds[2]:bounds[3]]
# DEM
dem = 'Buscot.dem.asc'
src = gdal.Open( dem )
dem = src.GetRasterBand(1).ReadAsArray()
src = None

# GLUE results
folder = 'Grid_060622_qvar146_U2'
S, A, B, C, r_ch, r_fp = read_flood_results( folder, read_maps=False )
F = (A-B)/(A+B+C) # F-score
Fs = ( F - F.min() )/( F.max() - F.min() )

# Filter
filter = F > 0.5
Fs_filtered = Fs[ filter ]
r_ch_filtered = r_ch[ filter ]
r_fp_filtered = r_fp[ filter ]
w_filtered = Fs_filtered/Fs_filtered.sum()
w = np.zeros( F.shape )
w[ filter ] = w_filtered

# Mean parameters
r_glue_mean = ( np.average(r_ch,weights=w), np.average(r_fp,weights=w) )
# MAP parameters
max_ix = np.where( F==F.max() )[0][0]
r_glue_max = ( r_ch[max_ix], r_fp[max_ix] )

## ==============================================
# 3. SIMULATION OF TIMES SERIES

# Flood maps output
sim_name = '110423_T10000N1_GLUE'
output_dir = os.path.join('simulations', sim_name)
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# Simulation period
T = 10000 # years

# Simulate inter-arrival times
rdT0 = m_pred.rvs( size=int( T*2*m_mean ) )
rdT = np.delete( rdT0, np.where(rdT0.cumsum() > T) ) # Trim at T years
rT = np.cumsum(rdT)
T0 = rT[-1]
N0 = rT.size

# Simulate discharges and amplitudes for each event

# Simulations of discharges using best-fit GPD
qi_best = st.genpareto.rvs( size=N0, *t_mle ) + u

# Posterior simulations of discharges
q_indices = np.random.choice( Xstack.shape[0], size=N0 )

# Posterior simulations of roughness parameters
glue_indices = np.random.choice( np.size(w), size=N0, p=w )

inds = []
rs = np.zeros( [N0, z.shape[0], z.shape[1]] )
S_sim = np.zeros( [N0, z.shape[0], z.shape[1]] )
S_sim_1 = np.zeros( [N0, z.shape[0], z.shape[1]] )
S_sim_2 = np.zeros( [N0, z.shape[0], z.shape[1]] )
S_sim_3 = np.zeros( [N0, z.shape[0], z.shape[1]] )
qi = np.zeros( N0 )
hi = np.zeros( N0 )
for i in range( N0 ):
    # Discharge simulation
    t = Xstack[ q_indices[i], : ]
    qi[i] = st.genpareto.rvs( t[0], loc=0, scale=t[1], size=1 )[0] + u
    
    # Lisflood simulation
    output = os.path.join( output_dir, sim_name + '_q{:.2f}.max'.format(qi[i]) )
    output_best = os.path.join( output_dir, sim_name + 'best_q{:.2f}.max'.format(qi[i]) )
    # Roughness parameters posterior simulation
    params_post_i = ( r_ch[glue_indices[i]], r_fp[glue_indices[i]] )
    
    # 1. Predictive posterior discharges + posterior flood simulator
    S_aux = Lisflood( 'array', qi[i], 68.43, params_post_i,
                        output=output )
    S_sim[i] = S_aux[bounds[0]:bounds[1],bounds[2]:bounds[3]]
    
    # 2. Predictive posterior discharges + MAP flood simulator
    S_aux = Lisflood( 'array', qi[i], 68.43, r_glue_max,
                            output=None )
    S_sim_1[i] = S_aux[bounds[0]:bounds[1],bounds[2]:bounds[3]]
    
    # 3. Best-fit GPD discharges + posterior flood simulator
    S_aux = Lisflood( 'array', qi_best[i], 68.43, params_post_i,
                            output=None )
    S_sim_2[i] = S_aux[bounds[0]:bounds[1],bounds[2]:bounds[3]]
    
    # 4. Best-fit GPD discharges + MAP flood simulator
    S_aux = Lisflood( 'array', qi_best[i], 68.43, 
                            r_glue_max, output=None )
    S_sim_3[i] = S_aux[bounds[0]:bounds[1],bounds[2]:bounds[3]]

    inds.append(i)

# Save/Load pickle
parameters = { 'rch':r_ch[glue_indices], 'rfp':r_fp[glue_indices],
               'q':qi, 'q_best':qi_best, 'r_glue_max': r_glue_max,
               'bounds':bounds, 'N0':N0, 'T0':T0 }

# with open( sim_name+'.pkl', 'wb') as file:
#     pickle.dump( parameters, file )
#     pickle.dump( S_sim, file)
#     pickle.dump( S_sim_1, file)
#     pickle.dump( S_sim_2, file)
#     pickle.dump( S_sim_3, file)
