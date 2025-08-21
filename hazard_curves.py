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
SIMULATION OF HAZARD CURVES FOR POSTERIOR PARAMETERS SAMPELS

This script aims to compute a hazard curve (for defined return periods) for different 
samples of model's parameters. It allows to compute the predictive posterior estimate
of the hazard, but also its confidence bounds.

1. Discharge frequency model: Bayesian POT approach from daily discharge historical data.
2. Flood simulator calibration: roughness parameters and their posterior distribution are 
    calibrated using GLUE.
3. Hazard curves calculations: For each sample of GPD parameters and roughness parameters
    it computes the hazard curve for a list of return periods.

"""


## =============================================================================
# 1. DISCHARGE FREQUENCY MODEL

# Read discharge data
filename = 'data/Buscot-daily_discharge.csv'
df = pd.read_csv( filename, usecols=[0,1], skiprows=19, parse_dates=[0] )

# Add 'time' and 'obs' index columns
df['obs'] = df['discharge']
df['time'] = np.arange( 0, len(df), 1 )

# Fill NaN values with interpolation
# df0 = df.interpolate('linear')

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
# diagnostic_plots( ti, distr, ti_params );

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
# SIMULATION OF TIMES SERIES

# Flood maps output
sim_name = '180423_100epis_GLUE'
output_dir = os.path.join('simulations', sim_name)
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# Number of hazard curves to simulate
N = 100

# Simulation return periods
Trs = np.array([1,2,5,10,25,50,100,250,500,750,1000])

# Posterior simulations of discharges
q_indices = np.random.choice( Xstack.shape[0], size=N )

# Posterior simulations of roughness parameters
glue_indices = np.random.choice( np.size(w), size=N, p=w )

qis = np.zeros([N, len(Trs)])
qis_best = np.zeros([N, len(Trs)])
S = np.zeros( [N, len(Trs), z.shape[0], z.shape[1]] )
Sbest = np.zeros( [N, len(Trs), z.shape[0], z.shape[1]] )
Sbest2 = np.zeros( [N, len(Trs), z.shape[0], z.shape[1]] )
Sbest3 = np.zeros( [N, len(Trs), z.shape[0], z.shape[1]] )
for j in range(N):

    # Current GPD parameters
    t = Xstack[ q_indices[j], : ]
    
    # Compute discharges for selected return periods
    Gis = 1/Trs/m_dist.mean()
    qis[j] = st.genpareto.ppf( 1-Gis, t[0], loc=0, scale=t[1] ) + u
    # ...for best-fit GPD parameters
    qis_best[j] = st.genpareto.ppf( 1-Gis, *t_mle ) + u

    # Current roguhness parameters
    params_post_i = ( r_ch[glue_indices[j]], r_fp[glue_indices[j]] )

    # Simulate inundation for each return period
    inds = []
    for i in range( len(Trs) ):

        # Output H simulation
        hi = 68.43
        # Lisflood simulation
        output = os.path.join( output_dir, sim_name + '_q{:.2f}.max'.format(qis[j][i]) )

        # 1. Predictive posterior discharges + posterior flood simulator
        S_aux = Lisflood( 'array', qis[j][i], hi, params_post_i, output=output )
        S[j,i] = S_aux[bounds[0]:(bounds[1]+1),bounds[2]:(bounds[3]+1)]
        
        # 2. Predictive posterior discharges + MAP flood simulator
        Sbest_aux = Lisflood( 'array', qis[j][i], hi, r_glue_max,
                        output=None )
        Sbest[j,i] = Sbest_aux[bounds[0]:(bounds[1]+1),bounds[2]:(bounds[3]+1)]
        
        # 3. Best-fit GPD discharges + posterior flood simulator
        Sbest2_aux = Lisflood( 'array', qis_best[j][i], hi, params_post_i,
                                output=None )
        Sbest2[j,i] = Sbest2_aux[bounds[0]:(bounds[1]+1),bounds[2]:(bounds[3]+1)]
        
        # 4. Best-fit GPD discharges + MAP flood simulator
        Sbest3_aux = Lisflood( 'array', qis_best[j][i], hi, 
                                r_glue_max, output=None )
        Sbest3[j,i] = Sbest3_aux[bounds[0]:(bounds[1]+1),bounds[2]:(bounds[3]+1)]


# Save/Load pickle
parameters = { 'rch':r_ch[glue_indices], 'rfp':r_fp[glue_indices],
               'q':qis, 'r_glue_max': r_glue_max,
               'bounds':bounds, 'N':N }

# with open( sim_name+'.pkl', 'wb') as file:
#     pickle.dump( parameters, file )
#     pickle.dump( S, file)
#     pickle.dump( Sbest, file)
#     pickle.dump( Sbest2, file)
#     pickle.dump( Sbest3, file)
