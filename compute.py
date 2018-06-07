import numpy as np
from pylab import *
import bokeh.plotting as plt
from bokeh.models import LinearColorMapper, LogTicker, ColorBar
import os, re
from scipy.interpolate import griddata
import scipy as scipy
import math
import matplotlib as mtplt
import matplotlib.cm as cm

def load_surfer(fname, fmt='ascii'):
    """
    Read a Surfer grid file and return three 1d numpy arrays and the grid shape

    Surfer is a contouring, gridding and surface mapping software
    from GoldenSoftware. The names and logos for Surfer and Golden
    Software are registered trademarks of Golden Software, Inc.

    http://www.goldensoftware.com/products/surfer

    According to Surfer structure, x and y are horizontal and vertical
    screen-based coordinates respectively. If the grid is in geographic
    coordinates, x will be longitude and y latitude. If the coordinates
    are cartesian, x will be the easting and y the norting coordinates.

    WARNING: This is opposite to the convention used for Fatiando.
    See io_surfer.py in cookbook.

    Parameters:

    * fname : str
        Name of the Surfer grid file
    * fmt : str
        File type, can be 'ascii' or 'binary'

    Returns:

    * x : 1d-array
        Value of the horizontal coordinate of each grid point.
    * y : 1d-array
        Value of the vertical coordinate of each grid point.
    * grd : 1d-array
        Values of the field in each grid point. Field can be for example
        topography, gravity anomaly etc
    * shape : tuple = (ny, nx)
        The number of points in the vertical and horizontal grid dimensions,
        respectively

    """
    assert fmt in ['ascii', 'binary'], "Invalid grid format '%s'. Should be \
        'ascii' or 'binary'." % (fmt)
    if fmt == 'ascii':
        # Surfer ASCII grid structure
        # DSAA            Surfer ASCII GRD ID
        # nCols nRows     number of columns and rows
        # xMin xMax       X min max
        # yMin yMax       Y min max
        # zMin zMax       Z min max
        # z11 z21 z31 ... List of Z values
        with open(fname) as ftext:
            # DSAA is a Surfer ASCII GRD ID
            id = ftext.readline()
            # Read the number of columns (nx) and rows (ny)
            nx, ny = [int(s) for s in ftext.readline().split()]
            # Read the min/max value of x (columns/longitue)
            xmin, xmax = [float64(s) for s in ftext.readline().split()]
            # Read the min/max value of  y(rows/latitude)
            ymin, ymax = [float64(s) for s in ftext.readline().split()]
            # Read the min/max value of grd
            zmin, zmax = [float64(s) for s in ftext.readline().split()]
            data = np.fromiter((float64(i) for line in ftext for i in
                                   line.split()), dtype='float64')
        # Create x and y numpy arrays
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
        x, y = [tmp.ravel() for tmp in np.meshgrid(x, y)]
    if fmt == 'binary':
        raise NotImplementedError(
            "Binary file support is not implemented yet.")
    return x, y, data, (ny,nx),zmin,zmax


def computePeak(filenames,MinMax,sigma):

    print 'compute:',MinMax
    print 'compute:',sigma

    x = []
    y = []
    grd = []
    nx = []
    ny = []
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    dx = []
    dy = []

    for file in filenames:

        print file

        [x_tmp,y_tmp,grd_tmp,(nx_tmp,ny_tmp),zmin_tmp,zmax_tmp]  = load_surfer(file)

        x.append(x_tmp)
        y.append(y_tmp)
        idx = [i for i,v in enumerate(grd_tmp) if v <= zmax_tmp]
        idx_nan = list(set(range(len(grd_tmp))) - set(idx))
        grd_tmp[idx_nan] = float(nan)
        grd.append(grd_tmp)
        nx.append(nx_tmp)
        ny.append(ny_tmp)
        xmin.append(x_tmp[0])
        xmax.append(x_tmp[-1])
        ymin.append(y_tmp[0])
        ymax.append(y_tmp[-1])
        dx.append(x_tmp[1]-x_tmp[0])
        dy.append(y_tmp[ny_tmp]-y_tmp[0])


    x1 = (np.max(xmin)).astype(float64)
    x2 = (np.min(xmax)).astype(float64)

    y1 = (np.max(ymin)).astype(float64)
    y2 = (np.min(ymax)).astype(float64)


    dxmax = np.max(dx).astype(float64)
    dymax = np.max(dy).astype(float64)


    ncols = (np.ceil( ( x2-x1) / dxmax )).astype(int)+1
    nrows = (np.ceil( ( y2-y1) / dymax )).astype(int)+1

    xnew = np.linspace(x1,x2,ncols)
    ynew = np.linspace(y1,y2,nrows)

    dxnew = xnew[1]-xnew[0]
    dynew = ynew[1]-ynew[0]


    X_new,Y_new = meshgrid(xnew,ynew)

    data_new = []

    validdata = []

    validdataAll = np.ones_like(X_new, dtype=bool)

    for i in range(len(grd)):

        print MinMax[i]

        if ( MinMax[i] == 'Min' ):

             data_coeff = -1.0

        else:

             data_coeff = 1.0

        data_new.append(griddata(np.array([x[i].ravel(),y[i].ravel()]).T,
                         data_coeff*grd[i].ravel(),(X_new,Y_new),
                         fill_value=nan, method='nearest'))


        validdata.append(np.logical_not(isnan(data_new[i])))
        validdataAll[:,:] = np.logical_and(validdata[i],validdataAll[:,:])


    mean_data = []
    std_data = []

    for i in range(len(grd)):

        mean_tmp = sum((data_new[i][validdataAll]).ravel())/np.sum(validdataAll.ravel())
        mean_data.append(mean_tmp)
        std_data.append(np.std(data_new[i][validdataAll]))



    std_coeff = np.linspace(1.0,2.0,10)

    fix_idx_opt = 0.0

    maskBoth = np.ones_like(X_new, dtype=bool)
    maskOr = np.ones_like(X_new, dtype=bool)
    data_check = np.ones_like(X_new, dtype=float64)


    maskBothOpt = np.ones_like(X_new, dtype=bool)



    fit_idx_opt = 0.0
    areaOpt = 0.0

    data_check[0:nrows,0:ncols] = data_new[0]
    data_check[isnan(data_new[0])] = -1.e10

    maskBoth = data_check > ( mean_data[0] + float(sigma[0])*std_data[0] )
    maskOr[0:nrows,0:ncols] = maskBoth[0:nrows,0:ncols]

    for j in range(1,len(grd)):

        data_check[0:nrows,0:ncols] = data_new[j]
        data_check[isnan(data_new[j])] = -1.e10

        dataMask = data_check > ( mean_data[j] + float(sigma[j])*std_data[j] )

        maskBoth = np.logical_and( dataMask , maskBoth )
        maskOr = np.logical_or( dataMask , maskOr )


    fit_idx = (np.sum(maskBoth)).astype(float64) / np.sum(maskOr).astype(float64)

    maskBothOpt[0:nrows,0:ncols] = maskBoth[0:nrows,0:ncols]
    areaOpt = np.sum(maskBoth)*dxnew*dynew

    print 'fitting index ',fit_idx
    print 'common peak region area',areaOpt

    # plot the maps

    origin = 'lower'

#    for i in range(len(grd)):
#
#        plt.figure()
#
#        best_value = mean_data[i] + std_coeff[max_fit_idx]*std_data[i]
#
#        CS = plt.contourf(X_new, Y_new, data_new[i], 50,
#                      cmap=plt.cm.bone,
#                      origin=origin)
#
#        CS2 = plt.contour(CS, levels=[best_value],
#                      colors='r',
#                      origin=origin)


    # minimum of normalized data

    data_masked = np.zeros_like(X_new, dtype=float64)

    data_norm = []

    for i in range(len(grd)):

        data_norm.append( (data_new[i].ravel()-mean_data[i])/std_data[i] )

    data_norm_stack = np.vstack([data_normi.T for data_normi in data_norm])

    data_masked = np.reshape(np.amin(data_norm_stack,axis=0),(nrows,ncols))

    data_masked[maskBothOpt==0] = -1.0

    data_masked[validdataAll==0] = float(nan)



#    plt.figure()
#    CS3 = plt.contourf(X_new, Y_new, data_masked, 50,
#                  cmap=plt.cm.jet,
#                  origin=origin)
#
#    cbar = plt.colorbar(CS3)



#    plt.figure()
#    plt.imshow(maskBothOpt, aspect='auto', interpolation='none',
#               extent=extents(xnew) + extents(ynew), origin='lower')

    #plt.show()


    return [data_masked,validdataAll,x1,x2,y1,y2,areaOpt,fit_idx,mean_data,std_data]


def compute(filenames,MinMax,sigma):

    np.set_printoptions(threshold='nan')

    [data_masked,validdataAll,x1,x2,y1,y2,areaOpt,fit_idx,mean_data,std_data] = computePeak(filenames,MinMax,sigma)


    ncols = data_masked.shape[1]
    nrows = data_masked.shape[0]
    header = "DSAA\n"
    header += str(ncols)+" "+str(nrows)+"\n"
    header += str(x1)+" "+str(x2)+"\n"
    header += str(y1)+" "+str(y2)+"\n"
    header += str(np.min(data_masked[validdataAll]))+" "+str(np.max(data_masked[validdataAll]))

    # create a new plot with a title and axis labels
    TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select,lasso_select"


    p = plt.figure(title="", tools=TOOLS,toolbar_location="above",
                   x_range=(x1, x2), y_range=(y1, y2))

    colormap =cm.get_cmap("jet") #choose any matplotlib colormap here
    bokehpalette = [mtplt.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]

    color_mapper = LinearColorMapper(palette=bokehpalette, low=-1, high=np.max(data_masked[validdataAll]))

    p.image(image=[data_masked], x=x1, y=y1, dw=x2-x1, dh=y2-y1, color_mapper=color_mapper,)

    color_bar = ColorBar(color_mapper=color_mapper,
                     label_standoff=12, border_line_color=None, location=(0,0))

    p.add_layout(color_bar, 'right')

    from bokeh.resources import CDN
    from bokeh.embed import components
    script, div = components(p)
    head = """
<link rel="stylesheet"
 href="http://cdn.pydata.org/bokeh/release/bokeh-0.12.4.min.css"
 type="text/css" />
<script type="text/javascript"
 src="http://cdn.pydata.org/bokeh/release/bokeh-0.12.4.min.js">
</script>
<script type="text/javascript">
Bokeh.set_log_level("info");
</script>
"""

    mean_data_np = np.absolute(np.array(mean_data))
    mean_data_str = np.array2string(mean_data_np, formatter={'float_kind':lambda mean_data_np: "%.2e" % mean_data_np})

    std_data_np = np.array(std_data)
    std_data_str = np.array2string(std_data_np, formatter={'float_kind':lambda std_data_np: "%.2e" % std_data_np})


    return head, script, div , data_masked , header , areaOpt , fit_idx , mean_data_str , std_data_str

if __name__ == '__main__':
    print compute(filename1='test1.dat',filename2='test2.dat')