## @ingroup Core
# Utilities.py
#
# Created:  Oct 2022, M. Clarke

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import numpy as np
 
def interp2d(x,y,xp,yp,zp,fill_value= None):
    """
    Bilinear interpolation on a grid. ``CartesianGrid`` is much faster if the data
    lies on a regular grid.
    Args:
        x, y: 1D arrays of point at which to interpolate. Any out-of-bounds
            coordinates will be clamped to lie in-bounds.
        xp, yp: 1D arrays of points specifying grid points where function values
            are provided.
        zp: 2D array of function values. For a function `f(x, y)` this must
            satisfy `zp[i, j] = f(xp[i], yp[j])`
    Returns:
        1D array `z` satisfying `z[i] = f(x[i], y[i])`.
    """
    #if xp.ndim != 1 or yp.ndim != 1:
        #raise ValueError("xp and yp must be 1D arrays")
    #if zp.shape != (xp.shape + yp.shape):
        #raise ValueError("zp must be a 2D array with shape xp.shape + yp.shape")

    ix = np.clip(np.searchsorted(xp, x, side="right"), 1, len(xp) - 1)
    iy = np.clip(np.searchsorted(yp, y, side="right"), 1, len(yp) - 1)

    # Using Wikipedia's notation (https://en.wikipedia.org/wiki/Bilinear_interpolation)
    z_11 = zp[ix - 1, iy - 1]
    z_21 = zp[ix, iy - 1]
    z_12 = zp[ix - 1, iy]
    z_22 = zp[ix, iy]

    z_xy1 = (xp[ix] - x) / (xp[ix] - xp[ix - 1]) * z_11 + (x - xp[ix - 1]) / (
        xp[ix] - xp[ix - 1]
    ) * z_21
    z_xy2 = (xp[ix] - x) / (xp[ix] - xp[ix - 1]) * z_12 + (x - xp[ix - 1]) / (
        xp[ix] - xp[ix - 1]
    ) * z_22

    z = (yp[iy] - y) / (yp[iy] - yp[iy - 1]) * z_xy1 + (y - yp[iy - 1]) / (
        yp[iy] - yp[iy - 1]
    ) * z_xy2

    if fill_value is not None:
        oob = np.logical_or(
            x < xp[0], np.logical_or(x > xp[-1], np.logical_or(y < yp[0], y > yp[-1]))
        )
        z = np.where(oob, fill_value, z)

    return z
