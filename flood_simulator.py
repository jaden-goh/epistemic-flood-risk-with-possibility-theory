import osgeo.gdal as gdal
import osgeo.osr as osr
import numpy as np
import pandas as pd
import csv, time
import subprocess, os
from shutil import copy

def read_flood_results( folder, read_maps=True ):

    # Scores
    df = pd.read_csv( os.path.join( folder, 'results.csv') )
    A = np.array(df['A'])
    B = np.array(df['B'])
    C = np.array(df['C'])
    r_ch = np.array(df['r_ch'])
    r_fp = np.array(df['r_fp'])
    df = None

    S = np.array([])
    if read_maps:
        # Read size of flood matrix
        csvfile = open( os.path.join( folder, 'results.csv'), "r")
        reader = csv.reader(csvfile, delimiter=',')
        Nsim = sum(1 for row in reader) - 1
        # Reset csv file reader
        csvfile.seek(0)
        
        # Dimension of flood matrix
        _ = next(reader)
        a = next(reader)
        src = gdal.Open( os.path.join( folder, a[-1]+'.max') )
        aux = src.GetRasterBand(1).ReadAsArray()
        # Reset csv file reader
        csvfile.seek(0)

        # Read 
        S = np.zeros([Nsim, aux.shape[0], aux.shape[1]])
        maxfiles = []
        i = 0
        _ = next(reader)
        for row in reader:
            print( 'Reading flood map {} of {}'.format(str(i+1),str(Nsim)) )
            maxfiles.append( row[-1] )
            src = gdal.Open( os.path.join( folder, maxfiles[i]+'.max') )
            S[i] = src.GetRasterBand(1).ReadAsArray()
            i += 1
            src = None
    
    return S, A, B, C, r_ch, r_fp


def wait_timeout(proc, seconds):
    """
    """
    start = time.time()
    end = start + seconds
    interval = min(seconds/1, .25)

    while True:
        result = proc.poll()
        if result is not None:
            return result
        if time.time() >= end:
            proc.kill()
            return -1
        time.sleep(interval)

def create_par(filename, files, title='', settings=[], **kwargs):
    """
    """
    # Create and write file
    par_file = open(filename,'w')

    # Title row
    par_file.write('# '+title+'\n')
    par_file.write('\n')

    # Optional key-word arguments
    for key, value in kwargs.items():
        par_file.write(key+"\t"+str(value)+'\n')

    # Files
    for key, value in files.items():
        par_file.write(key+"\t"+str(value)+'\n')
        
    # Single-word settings
    par_file.write('\n')
    for setting in settings:
        par_file.write(setting+'\n')

    # Close file
    par_file.close()
    
def create_bci(filename, bci):
    """
    """
    # Create and write file
    bci_file = open(filename,'w')
    # Boundary conditions
    for bc in bci:
        bci_file.write(bc[0]+'\t'+bc[1]+'\t'+bc[2]+'\t'+bc[3]+'\t'+bc[4]+'\n')
    # Close file
    bci_file.close()
    
def readLisfloodMax( filename, x ):
    src = gdal.Open( filename )
    array = src.GetRasterBand(1).ReadAsArray()
    z = np.array( [ array[x[i][0], x[i][1]] for i in range(len(x))] )
    return z, array

def Lisflood( x, q, h, theta, output=None ):
    """
    Returns the flood height for the set of points in x from a Lisflood run
    with parameters theta
    """
    
    # Input raster files
    files = {}
    files['DEMfile'] = 'Buscot.dem.asc'
    files['bcifile'] = 'Buscot.bci' 
    files['bdyfile'] = 'Buscot.bdy'
    files['SGCwidth'] = 'Buscot.width.asc'
    files['SGCbed'] = 'Buscot.bed.asc'
    files['SGCbank'] = files['DEMfile']
    files['startfile'] = 'Buscot.depth.asc'
    files['weirfile'] = 'Buscot.weir'

    # Settings
    options = {}
    options['resroot'] = 'temp'
    options['dirroot'] = 'temp'
    options['sim_time'] = 1382400
    options['initial_step'] = 1
    options['massint'] = 10*1
    options['saveint'] = options['sim_time']/9
    options['SGCn'] = str(theta[0])
    options['fpfric'] = str(theta[1])
    options['settings'] = ['mint_hk','elevoff']
    
    # Modify BCI file for different input discharges
    bci = [ ['E', '200000', '197650', 'HFIX', '68.43'],
          ['P','422972', '198081', 'QFIX', '1']
         ]
    bci[1][-1] = str(q/50)[:6] # modify input QFIX
    bci[0][-1] = str(h)[:6] # modify output HFIX
    create_bci( files['bcifile'], bci )
    
    # Create .par file
    parfile = 'Buscot.par'
    create_par(parfile, files, **options)

    # Run model
    opts = []
    os.environ["OMP_NUM_THREADS"] = str(4)
    # executable = './lisflood_linux' # for Linux
    executable = 'lisflood_intelRelease_double' # for Windows
    call = executable + ' ' + parfile
    proc = subprocess.Popen(call, shell=True)
    result = wait_timeout(proc, 120)

    # Save outputs
    source = os.path.join(options['dirroot'], options['resroot']+'.max')
    if output:
        copy(source, output)
    
    # Extract desired flood heights from .max file
    if x=='array':
        src = gdal.Open( source )
        return src.GetRasterBand(1).ReadAsArray()
    elif x:
        return readLisfloodMax( source, x.astype(int) )
    
def mask_arrays(base_array, mask_array, reverse=False):
    Xcount = np.size(base_array,1)
    Ycount = np.size(base_array,0)
    # Loop through base array
    new_array = np.zeros([Ycount,Xcount])
    count = 0
    for i in range(Ycount):
        for j in range(Xcount):
            chanmask_value = mask_array[i, j]
            if reverse:
                if chanmask_value > 0:
                    count += 1
                    new_array[i,j] = base_array[i,j]
            elif not reverse:
                if chanmask_value <= 0:
                    new_array[i,j] = base_array[i,j]

    return(new_array)

def mask_raster(base_raster, mask_raster, band1=1, band2=1, reverse=False):
    # Base raster
    base_transform = base_raster.GetGeoTransform()
    base_array = base_raster.GetRasterBand(band1).ReadAsArray()
    # Xcount = base_raster.RasterXSize # number of cols
    # Ycount = base_raster.RasterYSize # number of rows

    # Mask raster
    mask_transform = mask_raster.GetGeoTransform()
    mask_array = mask_raster.GetRasterBand(band2).ReadAsArray()

    return mask_arrays(base_array, mask_array, reverse=reverse)

def jaccard_fit(observed, predicted, chanmask, region_polygon=False,
                reverse=False, save_comparison=False):

    observed_source = gdal.Open(observed)
    predicted_source = gdal.Open(predicted)
    chanmask_source = gdal.Open(chanmask)

    # Remove chanmask
    observed_array = mask_raster(observed_source, chanmask_source, reverse=reverse)
    predicted_array = mask_raster(predicted_source, chanmask_source, reverse=reverse)

    # Mask with region polygon
    if region_polygon:
        mask_array = region_polygon.GetRasterBand(1).ReadAsArray()
        # Mask arrays
        observed_array = mask_arrays(observed_array, mask_array, reverse=True)
        predicted_array = mask_arrays(predicted_array, mask_array, reverse=True)
        # Flush temporary raster
        target_ds = None

    # Compare observed vs predicted maps
    correctly_predicted = 0
    over_predicted = 0
    under_predicted = 0
    comparison_array = np.zeros([np.size(observed_array,0),
                                 np.size(observed_array,1)])
    for i in range(np.size(observed_array,0)):
        for j in range(np.size(observed_array,1)):

            obs = observed_array[i,j]
            pred = predicted_array[i,j]
            # Correctly predicted
            if obs and pred:
                correctly_predicted += 1
                comparison_array[i,j] = 1
            # Over-predicted
            if pred and (not obs):
                over_predicted += 1
                comparison_array[i,j] = 2
            # Under-predicted
            if (not pred) and obs:
                under_predicted += 1
                comparison_array[i,j] = 3

    # Save comparison array in raster
    if save_comparison:
        cols = comparison_array.shape[1]
        rows = comparison_array.shape[0]

        driver = gdal.GetDriverByName( "GTiff" )
        outRaster = driver.Create(save_comparison, cols, rows, 1, gdal.GDT_Byte)
        outRaster.SetGeoTransform(observed_source.GetGeoTransform())
        outband = outRaster.GetRasterBand(1)
        outband.WriteArray(comparison_array)
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromWkt(observed_source.GetProjectionRef())
        outRaster.SetProjection(outRasterSRS.ExportToWkt())
        outband.FlushCache()

    # Results
    A = correctly_predicted
    B = over_predicted
    C = under_predicted
    F = (A-B)/(A+B+C)
    return {'A':A, 'B':B, 'C':C, 'F':F}
