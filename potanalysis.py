import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

def ecdf(data):
    """ Compute ECDF """
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / (n+1)
    return(x,y)

def autocorr( series, max_lag=20, plot=False, **kwargs ):
    acf = [series.autocorr(lag=i) for i in range(0,max_lag)]
    ax = None
    if plot:
        fig, ax = plt.subplots()
        ax.bar( range(0,max_lag), acf, **kwargs )
        ax.set_xlabel( 'lag' )
        ax.set_ylabel( 'autocorrelation' )
    return acf, ax

def bayesian_diagnostic_plots( xu, dist, params, conf=0.9, plot='Tr' ):
    
    # Simulate from the posterior
    Fi = np.zeros( [len(xu), params.shape[0]] )
    xu_o = np.sort( xu )
    for i in range( params.shape[0] ):
        t = params[i,:]
        Fi[:,i] = dist.cdf( xu_o, t[0], loc=0, scale=t[1] )
    # Return levels
    Tri = 1/(1-Fi)
    # Empirical
    Fi_emp = ( ecdf(xu_o)[1] )
    Tri_emp = 1/(1-Fi_emp)
    # Confidence intervals
    qi = (1 - conf)/2
    qs = (1 + conf)/2
    Fi_ci = np.quantile( Fi, [qi,qs], axis=1) 
    Fi_mean = np.mean( Fi, axis=1 )
    Tri_ci = np.quantile( Tri, [qi,qs], axis=1) 
    Tri_mean = np.mean( Tri, axis=1 )
    
    if plot:
        # p-p Plot
        fig, ax = plt.subplots( figsize=(12,4), nrows=1, ncols=2 )
        ax[0].fill_betweenx( Fi_emp, Fi_ci[0], Fi_ci[1], alpha=0.5,
                        label='Conf. interval: {}'.format(conf))
        ax[0].plot( Fi_mean, Fi_emp )
        ax[0].plot( [0,1], [0,1], color='red' )
        ax[0].set_xlabel( 'Theoretical cumulative distribution' )
        ax[0].set_ylabel( 'Empirical cumulative distribution' )
        ax[0].set_title( 'p-p plot' )
        
        # Return level plot
        ax[1].plot( Tri_emp, xu_o, 'bo' )
        ax[1].fill_betweenx( xu_o, Tri_ci[0], Tri_ci[1], alpha=0.5,
                        label='Conf. interval: {}'.format(conf))
        ax[1].plot( Tri_mean, xu_o )
        ax[1].set_xlabel( 'Return level' )
        ax[1].set_ylabel( 'Discharge [m/s3]' )
        ax[1].set_title( 'Return level plot' )
        # ax.set_yscale('log')
        ax[1].set_xscale('log')
        
        return Fi, Tri, fig
    
    return Fi, Tri

def diagnostic_plots( xu, dist, params ):
    fig, ax = plt.subplots( figsize=(8,4), nrows=1, ncols=2 )
    res = st.probplot( xu, dist=dist, sparams=params, plot=ax[0] )
    res = ppplot( xu, dist=dist, sparams=params, ax=ax[1] )
    return ax

def ppplot( x, dist, sparams, ax=None):
    th_cdf = dist.cdf( x, *sparams )
    th_cdf.sort()
    _, e_cdf = ecdf( x )

    if ax:
        ax.plot( th_cdf, e_cdf, 'o', color='b', markersize=6 )
        ax.plot( [0,1], [0,1], color='red' )
        ax.set_xlabel( 'Theoretical cumulative distribution' )
        ax.set_ylabel( 'Empirical cumulative distribution' )
        ax.set_title( 'p-p plot' )
    
    return ax

def fit_ci( x, dist, Nboot=500, **kwargs ):
    """
    Fit continuous distribution to data, and obtain confidence intervals for
    the parameters using non-parametric bootstrap.
    """
    
    # Create bootstraps series sampling with replacement from x
    bt = np.random.choice( x, (len(x), Nboot) )
    
    # Compute MLE parameters
    *sh_mle, loc_mle, s_mle = dist.fit( x, **kwargs )
    
    # Fit distribution to each bootstrapped series
    shape = []
    location = []
    scale = []
    for i in range(Nboot):
        xi = bt[:,i]
        *sh, loc, s = dist.fit( xi, **kwargs )
        # Add to list of parameters
        shape.append( sh[0] )
        location.append( loc )
        scale.append( s )
        
    # Compute ccovariance matrix of parameters
    cov = np.cov( np.array([shape,location,scale]) )
    
    return (sh_mle[0], loc_mle, s_mle), cov

def TC( x, conf=0.9, lims=None, N=30, plot=False ):
    """
    Threshold choice model by computing the modified scale and shape parameters
    for different values of threshold u
    
    """
    if not lims:
        thresholds = np.linspace( np.min(x), np.sort(x)[-4], N )
    else:
        thresholds = np.linspace( lims[0], lims[1], N )
    scale_mod = []
    ci_scale = []
    shape = []
    ci_shape = []
    z = st.norm.ppf(conf) 
    for u in thresholds:
        # Points over threshold
        xu = x[x-u>0] - u
        # Fit GDP
        # sh, loc, scale = st.genpareto.fit( xu, floc=0 )
        mles, cov = fit_ci( xu, st.genpareto, Nboot=500, floc=0 )
        # Compute modified scale
        scale_mod.append( mles[2] - mles[0]*u )
        shape.append( mles[0] )
        # Compute CIs
        shape_var = np.sqrt(cov[0,0])
        ci_shape.append( shape_var * z )
        scale_var = np.sqrt(cov[0,0] * u**2 - 2*cov[0,2]*u + cov[2,2])
        ci_scale.append( scale_var * z )
    
    ax = []
    if plot:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].errorbar( thresholds, scale_mod, yerr=ci_scale, color=colors[0],
                        fmt='o' )
        ax[0].set_xlabel('threshold')
        ax[0].set_ylabel('modificed scale')
        ax[1].errorbar( thresholds, shape, yerr=ci_shape, color=colors[0],
                        fmt='o' )
        ax[1].set_xlabel('threshold')
        ax[1].set_ylabel('modified shape')
    
    return scale_mod, shape, ax

def MRL( x, lims=None, conf=0.95, N=1000, plot=False ):
    """
    Compute the mean residual life for data series with threshold 0<u<xmax
    
    args:
        - x:    List or numpy array of data
        - plot: Flag to plot
    
    """
    if not lims:
        thresholds = np.linspace( np.min(x), np.sort(x)[-4], N )
    else:
        thresholds = np.linspace( lims[0], lims[1], N )
    mrl = []
    mrl_u = []
    mrl_l = []
    for u in thresholds:
        # Points over threshold
        xu = x[x-u>0] - u
        # Sample mean and error
        m = xu.mean()
        s = m/np.sqrt(len(xu))
        # Save output vars
        mrl.append( m )
        mrl_u.append( st.norm.ppf( 1/2+conf/2, m, s ) )
        mrl_l.append( st.norm.ppf( 1/2-conf/2, m, s ) )
    
    ax = []
    if plot:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig, ax = plt.subplots()
        ax.plot( thresholds, mrl, color=colors[0] )
        ax.plot( thresholds, mrl_u, linestyle='--', color=colors[0] )
        ax.plot( thresholds, mrl_l, linestyle='--', color=colors[0] )
        
    return thresholds, (mrl, mrl_u, mrl_l), ax

def clust( df, u, time_cond, clust_max=True, plot=False):
    """
    
    args:
        - x: Dataset (list or numpy array). Needs 'obs' and 'time' columns. 
        - u: threshold. Should be in the same units than x.
        - time_cond: Time condition to ensure independence between events.
                     Should be defined in numbers of data points.
        - clust_max: [bool] If False a list containing the clusters exceedances
                     is returned. If True, only each cluster's maxima is returned
        - plot:
    
    The clusters of exceedances are defined as follows:
        - The first exceedance initiates the first cluster
        - The first observation under the threshold u “ends” the current cluster
             unless time_cond does not hold
        - The next exceedance initiates a new cluster
        - The process is iterated as needed.
    
    """
    ################### POSSIBILITY EDITS ####################################
    def record_length_years(df):
        if 'time' in df.columns and np.issubdtype(df['time'].dtype, np.number):
            return float(df['time'].iloc[-1] - df['time'].iloc[0]) / 365.25
        elif 'date' in df.columns:
            t = pd.to_datetime(df['date'])
            return (t.max() - t.min()).days / 365.25
        else:
            return len(df) / 365.25
    ###########################################################################

    # Keep only exceedances
    dfu = df[ df['obs'] > u ]
    
    # Create column with time differences between exceedances
    dfu['dt'] = np.append( 0, np.diff( dfu['time'] ) )
        
    # Beginning of clusters rows
    cl_ind = dfu[ dfu['dt'] > time_cond ].index
    cl_ind = cl_ind.union( pd.Index( [dfu.index[0]] ) ) # Add first cluter
    
    # Number of clusters
    Nclust = cl_ind.shape[0]
    
    # Plot
    ax = []
    if plot:
        fig, ax = plt.subplots(figsize=(8.5/2.54,5/2.54))
        # plt.subplots_adjust( left=0.07, bottom=0.16, right=0.97, top=0.94 )
        fig.tight_layout()
        df.plot( x='date', y='obs', ax=ax, x_compat=True, color='gray', legend=False )
        # ax.xaxis.set_major_locator( mdates.YearLocator()) # Poner labels solo en cada año
        # ax.xaxis.set_major_formatter( mdates.DateFormatter("%Y") )
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        ax.set_xlabel( 'date' )
        ax.set_ylabel( 'discharge [m/s3]' )
        ax.tick_params( axis='x' )
        ax.tick_params( axis='y' )
        # ax.set_xlim([-100,7000])
        # ax.set_title( 'Declustering' )
        ax.axhline( y=u, linewidth=0.5, color='k')
    
    # Maxima of each cluster
    cluster_out = np.array([])
    index_out = np.array([], 'int64')
    for i in range(Nclust-1):
        ci = cl_ind[i]
        cf = cl_ind[i+1] - 1
        cluster = dfu.loc[ci:cf]
        cluster_max = cluster['obs'].max()
        cluster_idxmax = cluster['obs'].idxmax()
        # Append to outputs
        cluster_out = np.append( cluster_out, cluster_max )
        index_out = np.append( index_out, cluster_idxmax )
        # Plot
        if plot:
            ax.plot( cluster['date'], cluster['obs'], color='r', linewidth=0.5 )
            ax.fill_between( cluster['date'], y1=cluster['obs'], y2=u, 
                             color='gray', alpha=0.6)
            ax.plot( df.loc[cluster_idxmax]['date'], cluster_max, 'x', color='b' )
    
    return df.loc[index_out], ax

## =============================================================================
if __name__ == '__main__':

    # Dataset
    filename = 'data//ardieres.csv'

    # Load data frame
    df = pd.read_csv( filename )
    df = df.dropna()
    
    # Plot time series
    fig, ax = plt.subplots()
    ax.plot( df['obs'], linewidth=0.5 )

    # Autocorrelation of the series
    acf, ax = autocorr( df['obs'], plot=True )

    # Declustering of series
    df_cl, ax = clust( df, u=0.85, time_cond=7/365, clust_max=True, plot=True)
    acf, ax = autocorr( df_cl['obs'], plot=True )
    
    ## Define threshold
    # Mean Residual Life Plot
    _, mrl, ax = MRL( df_cl['obs'], lims=(0,18), plot=True )

    # Modified scale and shape plots
    scale_mod, shape, ax = TC( df_cl['obs'], N=25, lims=(0,18), plot=True )
    
    # Fit GPD to exceedance points with selected threshold u
    u = 6
    df_cl, _ = clust( df, u=u, time_cond=7/365, clust_max=True, plot=False)
    x = df_cl['obs'][ df_cl['obs']>u ] - u
    fitted_args = st.genpareto.fit( x, floc=0 )

    # Diagnostics plots (qq and pp plots)
    diagnostic_plots( x, st.genpareto, fitted_args )

    plt.show()
