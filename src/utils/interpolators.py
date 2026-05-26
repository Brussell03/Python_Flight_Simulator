import numpy as np

def fastInterp1(x, y, xi):
    """
    Performs linear interpolation for a given data set.

    Args:
        x: Array of independent data points.
        y: Array of dependent data points.
        xi: Scalar value to interpolate.

    Returns:
        yi: Interpolated dependent data value for xi.
    """

    # Ensure x is a column vector
    x = x.reshape(-1, 1) 

    # Check if lengths of x and y match
    if len(x) != len(y):
        raise ValueError("Lengths of x and y must be equal.")

    # Create intervals array (updated 10/28)
    iia = np.array([
        [ -np.inf, x[0, 0], y[0], y[0], 1],
        *np.vstack([x[:-1, 0], x[1:, 0], y[:-1], y[1:], 2 * np.ones(len(x) - 1)]).T,
        [x[-1, 0], np.inf, y[-1], y[-1], 3]]) 

    # Find the relevant interval for xi
    xyc = iia[ (xi > iia[:, 0]) & (xi <= iia[:, 1]) , :]

    # Extract data from the interval
    x0 = xyc[0,0] 
    x1 = xyc[0,1]
    y0 = xyc[0,2]
    y1 = xyc[0,3]
    ic = xyc[0,4]

    # Perform interpolation based on the code
    if ic == 2:
        yi = (y0 * (x1 - xi) + y1 * (xi - x0)) / (x1 - x0)
    elif ic == 1:
        yi = y0
    elif ic == 3:
        yi = y1
    else:
        raise RuntimeError("Interpolation failure.")

    return yi

def fastInterp2(x, y, z, xi, yi):
    """
    fastInterp2(x, y, z, xi, yi) is a bilinear interpolator 
    that takes in x and y as vectors, z as a matrix, and xi and yi as 
    scalar independent data values to determine the dependent data value, zi.

    x - vector of length n
    y - vector of length m
    z - 2D numpy array of size n by m 

    This interpolator performs a linear interpolation within the domain of 
    data x and y. If xi, yi is outside this domain, the value of z(x,y) at the
    nearest neighboring (x,y) point is assigned.
    
    This bilinear interpolator uses the weighted mean approach.
    """
    
    # Ensure x and y are column vectors (converts to 2D arrays)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1) 
    
    # Get number of rows (nz) and columns (mz), and lengths of x and y
    nz = z.shape[0]
    mz = z.shape[1]
    lx = len(x)
    ly = len(y)
    
    # Check if lengths of x and y match z
    if lx != nz:
        raise ValueError("Length of x and number of rows in z must be equal.")
    
    if ly != mz:
        raise ValueError("Length of y and number of columns in z must be equal.")
    
    # Create interval arrays
    ix = np.array([
        [ -np.inf, x[0, 0], 1],
        *np.vstack([x[:-1, 0], x[1:, 0], 2*np.ones(lx - 1)]).T,
        [x[-1, 0], np.inf, 3]]) 
    
    iy = np.array([
        [ -np.inf, y[0, 0], 1],
        *np.vstack([y[:-1, 0], y[1:, 0], 2*np.ones(ly - 1)]).T,
        [y[-1, 0], np.inf, 3]]) 
    
    # Construct logical vectors to select appropriate x, y, and z data
    x_logic = (xi > ix[:, 0]) & (xi <= ix[:, 1])
    y_logic = (yi > iy[:, 0]) & (yi <= iy[:, 1])
    
    x_logic_shifted = np.roll(x_logic, -1)
    y_logic_shifted = np.roll(y_logic, -1)
    
    # Find the relevant interval for xi and yi
    xic = ix[ x_logic , :]
    x0 = xic[0,0]
    x1 = xic[0,1]
    
    yic = iy[ y_logic , :]
    y0 = yic[0,0]
    y1 = yic[0,1]

    # Specify indices to select appropriate z data
    if (xic[0,2] == 2) & (yic[0,2] == 2):
        
        # Get x and y indices in z that bound the data point xi, yj
        i   = np.where(x_logic)[0][0]
        im1 = i-1
        j   = np.where(y_logic)[0][0]
        jm1 = j-1
        
        # Get z at all four vertices
        z0 = z[im1,   j]
        z1 = z[  i,   j]
        z2 = z[im1, jm1]
        z3 = z[  i, jm1]
        
        # Compute denominator of N coefficients
        den = (x1 - x0)*(y1 - y0)

        # Compute N coefficients
        Na = (x1 - xi)*(yi - y0)/den
        Nb = (xi - x0)*(yi - y0)/den
        Nc = (x1 - xi)*(y1 - yi)/den
        Nd = (xi - x0)*(y1 - yi)/den

        # Interpolate
        zi = z0*Na + z1*Nb + z2*Nc + z3*Nd
        
    elif (xic[0,2] == 1) & (yic[0,2] == 1):
        
        # Select corner point as point closest to (xi,yi)
        zi = z[0,0]
        
    elif (xic[0,2] == 1) & (yic[0,2] == 2):
        
        # Find y index that corresponds to closest point
        y_idx = nnidx(y0, y1, yi, ly, y_logic)
        
        # Select z value closest to point (xi,yi)
        zi = z[0, y_idx].item()
        
    elif (xic[0,2] == 1) & (yic[0,2] == 3):
        
        # Select corner point as point closest to (xi,yi)
        zi = z[0, -1].item()
        
    elif (xic[0,2] == 2) & (yic[0,2] == 1):
        
        # Find x index that corresponds to closest point
        x_idx = nnidx(x0, x1, xi, lx, x_logic)
        
        # Select z value closest to point (xi,yi)
        zi = z[x_idx, 0].item()
        
    elif (xic[0,2] == 2) & (yic[0,2] == 3):
        
        # Find y index that corresponds to closest point
        x_idx = nnidx(x0, x1, xi, lx, x_logic)
        
        # Select z value closest to point (xi,yi)
        zi = z[x_idx, -1].item()
        
    elif (xic[0,2] == 3) & (yic[0,2] == 1):
        
        # Select corner point as point closest to (xi,yi)
        zi = z[-1, 0].item()
        
    elif (xic[0,2] == 3) & (yic[0,2] == 2):
        
        # Find y index that corresponds to closest point
        y_idx = nnidx(y0, y1, yi, ly, y_logic)
        
        # Select z value closest to point (xi,yi)
        zi = z[-1, y_idx].item()
        
    elif (xic[0,2] == 3) & (yic[0,2] == 3):
        
        # Select corner point as point closest to (xi,yi)
        zi = z[-1, -1].item()
        
    else:
        
        print('2D interpolator error: incorrect or missing boundary code')

    return zi

def nnidx(p0, p1, p_i, lp, p_logic):

    # Compute distances of p0 and p1 to p_i
    pdist = abs( np.array([[p0, p1]]) - p_i )

    # Tie-breaking: choose the lower index if equidistant
    closest_index = np.argmin(pdist, axis=1) 

    # Combine indices of y values
    idx = np.arange(lp+1)
    idx = idx[p_logic][0]
    idx = np.array([idx-1, idx])

    # Select y index that corresponds to closest point
    idx = idx[closest_index]

    return idx