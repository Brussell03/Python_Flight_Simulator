import numpy as np
from numba import njit

@njit
def fastInterp1(x, y, xi):
    """
    Numba-optimized 1D linear interpolation.
    Assumes x and y are flat 1D numpy arrays of equal length.
    """
    if xi <= x[0]:
        return y[0]
    elif xi >= x[-1]:
        return y[-1]
    
    # searchsorted finds index i such that x[i-1] < xi <= x[i]
    i = np.searchsorted(x, xi, side='left')
    
    x0, x1 = x[i-1], x[i]
    y0, y1 = y[i-1], y[i]
    
    return (y0 * (x1 - xi) + y1 * (xi - x0)) / (x1 - x0)


@njit
def get_closest_idx(arr, val, i):
    """Helper function to find nearest neighbor index without array allocations."""
    if abs(val - arr[i-1]) <= abs(arr[i] - val):
        return i - 1
    return i

@njit
def fastInterp2(x, y, z, xi, yi):
    """
    Numba-optimized bilinear interpolation with custom nearest-neighbor boundary logic.
    Assumes x and y are flat 1D arrays, z is a 2D array of shape (len(x), len(y)).
    """
    nx = len(x)
    ny = len(y)

    # Determine X status and interval index
    if xi <= x[0]:
        x_status = 1
        i = 1 
    elif xi >= x[-1]:
        x_status = 3
        i = nx - 1 
    else:
        x_status = 2
        i = np.searchsorted(x, xi, side='left')

    # Determine Y status and interval index
    if yi <= y[0]:
        y_status = 1
        j = 1
    elif yi >= y[-1]:
        y_status = 3
        j = ny - 1
    else:
        y_status = 2
        j = np.searchsorted(y, yi, side='left')

    # Status 2,2: Inside domain - Bilinear Interpolation
    if x_status == 2 and y_status == 2:
        x0, x1 = x[i-1], x[i]
        y0, y1 = y[j-1], y[j]
        
        z00 = z[i-1, j-1]
        z10 = z[i, j-1]
        z01 = z[i-1, j]
        z11 = z[i, j]
        
        den = (x1 - x0) * (y1 - y0)
        
        w00 = (x1 - xi) * (y1 - yi) / den
        w10 = (xi - x0) * (y1 - yi) / den
        w01 = (x1 - xi) * (yi - y0) / den
        w11 = (xi - x0) * (yi - y0) / den
        
        return z00 * w00 + z10 * w10 + z01 * w01 + z11 * w11

    # Corner Cases: Pure assignment
    elif x_status == 1 and y_status == 1:
        return z[0, 0]
    elif x_status == 1 and y_status == 3:
        return z[0, -1]
    elif x_status == 3 and y_status == 1:
        return z[-1, 0]
    elif x_status == 3 and y_status == 3:
        return z[-1, -1]
        
    # Edge Cases: Nearest Neighbor snapping
    elif x_status == 1 and y_status == 2:
        y_idx = get_closest_idx(y, yi, j)
        return z[0, y_idx]
    elif x_status == 3 and y_status == 2:
        y_idx = get_closest_idx(y, yi, j)
        return z[-1, y_idx]
    elif x_status == 2 and y_status == 1:
        x_idx = get_closest_idx(x, xi, i)
        return z[x_idx, 0]
    elif x_status == 2 and y_status == 3:
        x_idx = get_closest_idx(x, xi, i)
        return z[x_idx, -1]
    
    raise RuntimeError("2D interpolator error: unhandled boundary case.")