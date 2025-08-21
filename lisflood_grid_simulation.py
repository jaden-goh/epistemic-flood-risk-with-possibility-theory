from flood_simulator import Lisflood, jaccard_fit

import os, csv
import numpy as np


"""
LISFLOOD GRID SIMULATION

This script computes the inundation from Lisflood for a uniform grid of roughness parameters.
For each simulated inundation, it compares it with a flood extent observation and computes
the F-score (jaccard index). It saves all the inundation rasters, and the scoring for each
in a .csv.

"""

# Project name
project = 'Buscot_grid' # This will serve to name the input files for the .par file
resroot = 'Grid_0406_large' # This serves to name the output files
output_dir = 'results//'+resroot # This will save some of the output files to that folder ('results' folder should already exist)
 
# Input parameters
r_ch = np.arange( 0.001, 0.3, 0.002)
r_fp = np.arange( 0.001, 0.3, 0.01)
H = 68.43 # Fixed downstream water height
Q = 1.46 # Fixed input discharge

# Observations
observed = 'data//BuscotFlood92.tiff'

# Create list with all combinations
params = np.array(np.meshgrid(r_ch, r_fp)).T.reshape(-1,2)

# Initialize results csv
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
results_csv = os.path.join( output_dir, 'results.csv')
with open(results_csv, "w", newline='') as f:
    # Writers
    writer = csv.writer(f, delimiter=',', quotechar='"')
    # Headers
    writer.writerow(['r_ch', 'r_fp', 'A', 'B', 'C', 'F', 'resroot'])
    f.flush()

    count = 0
    for param in params:
        count += 1
        # Current run parameters
        r_ch = param[0]
        r_fp = param[1]

        # Save outputs
        outputs = ['max']
        new_name = resroot + '_sim' + str(count)
        predicted = os.path.join( output_dir, new_name+'.max' )

        # Run lisflood
        _ = Lisflood( 'array', Q, H, [r_ch, r_fp], output=predicted ) 
        
        # Compare with an observations map
        chanmask0 = 'Buscot.bed.asc'
        save_comparison = False
        fit = jaccard_fit( observed, predicted, chanmask0,
                           save_comparison=save_comparison )

        # Save metadata
        datarow = [str(r_ch), str(r_fp), str(fit['A'])[:9],
                   str(fit['B'])[:9], str(fit['C'])[:9], str(fit['F'])[:9],
                   new_name]
        writer.writerow(datarow)
        f.flush()


