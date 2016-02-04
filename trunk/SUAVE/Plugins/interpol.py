""" interpolate data given on an Nd rectangular grid, uniform or non-uniform.

Purpose: extend the fast N-dimensional interpolator
`scipy.ndimage.map_coordinates` to non-uniform grids, using `np.interp`.

Background: please look at
http://en.wikipedia.org/wiki/Bilinear_interpolation
http://stackoverflow.com/questions/6238250/multivariate-spline-interpolation-in-python-scipy
http://docs.scipy.org/doc/scipy-dev/reference/generated/scipy.ndimage.interpolation.map_coordinates.html

Example
-------
Say we have rainfall on a 4 x 5 grid of rectangles, lat 52 .. 55 x lon -10 .. -6,
and want to interpolate (estimate) rainfall at 1000 query points
in between the grid points.

        # define the grid --
    griddata = np.loadtxt(...)  # griddata.shape == (4, 5)
    lo = np.array([ 52, -10 ])  # lowest lat, lowest lon
    hi = np.array([ 55, -6 ])   # highest lat, highest lon

        # set up an interpolator function "interfunc()" with class Intergrid --
    interfunc = Intergrid( griddata, lo=lo, hi=hi )

        # generate 1000 random query points, lo <= [lat, lon] <= hi --
    query_points = lo + np.random.uniform( size=(1000, 2) ) * (hi - lo)

        # get rainfall at the 1000 query points --
    query_values = interfunc( query_points )  # -> 1000 values

What this does:
    for each [lat, lon] in query_points:
        1) find the square of griddata it's in,
            e.g. [52.5, -8.1] -> [0, 3] [0, 4] [1, 4] [1, 3]
        2) do bilinear (multilinear) interpolation in that square,
            using `scipy.ndimage.map_coordinates` .
Check:
    interfunc( lo ) -> griddata[0, 0],
    interfunc( hi ) -> griddata[-1, -1] i.e. griddata[3, 4]

Parameters
----------
    griddata: numpy array_like, 2d 3d 4d ...
    lo, hi: user coordinates of the corners of griddata, 1d array-like, lo < hi
    maps: a list of `dim` descriptors of piecewise-linear or nonlinear maps,
        e.g. [[50, 52, 62, 63], None]  # uniformize lat, linear lon
    copy: make a copy of query_points, default True;
        copy=False overwrites query_points, runs in less memory
    verbose: default 1: print a 1-line summary for each call, with run time
    order=1: see `map_coordinates`
    prefilter: 0 or False, the default: smoothing B-spline
              1 or True: exact-fit interpolating spline (IIR, not C-R)
              1/3: Mitchell-Netravali spline, 1/3 B + 2/3 fit
        (prefilter is only for order > 1, since order = 1 interpolates)

Non-uniform rectangular grids
-----------------------------
What if our griddata above is at non-uniformly-spaced latitudes,
say [50, 52, 62, 63] ?  `Intergrid` can "uniformize" these
before interpolation, like this:

    lo = np.array([ 50, -10 ])
    hi = np.array([ 60, -6 ])
    maps = [[50, 52, 62, 63], None]  # uniformize lat, linear lon
    interfunc = Intergrid( griddata, lo=lo, hi=hi, maps=maps )

This will map (transform, stretch, warp) the lats in query_points column 0
to array coordinates in the range 0 .. 3, using `np.interp` to do
piecewise-linear (PWL) mapping:
    50  51  52  53  54  55  56  57  58  59  60  61  62  63  # lo[0] .. hi[0]
    0   .5  1   1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2   3

`maps[1] None` says to map the lons in query_points column 1 linearly:
    -10  -9  -8  -7  -6  # lo[1] .. hi[1]
    0    1   2   3   4

More doc: https://denis-bz.github.com/docs/intergrid.html

"""
# split class Gridmap ?

from __future__ import division
from time import time
# warnings
import autograd.numpy as np 
from scipy.ndimage import map_coordinates, spline_filter

__version__ = "2013-07-01 jul denis"
__author_email__ = "denis-bz-py@t-online.de"  # comments welcome, testcases most welcome

#...............................................................................
class Intergrid:
    __doc__ = globals()["__doc__"]

    def __init__( self, griddata, lo, hi, maps=[], copy=True, verbose=1,
            order=1, prefilter=False ):
        griddata = np.asanyarray( griddata )
        dim = griddata.ndim  # - (griddata.shape[-1] == 1)  # ??
        assert dim >= 2, griddata.shape
        self.dim = dim
        if np.isscalar(lo):
            lo *= np.ones(dim)
        if np.isscalar(hi):
            hi *= np.ones(dim)
        assert lo.shape == (dim,), lo.shape
        assert hi.shape == (dim,), hi.shape
        self.loclip = lo = np.asarray_chkfinite( lo ).copy()
        self.hiclip = hi = np.asarray_chkfinite( hi ).copy()
        self.copy = copy
        self.verbose = verbose
        self.order = order
        if order > 1  and 0 < prefilter < 1:  # 1/3: Mitchell-Netravali = 1/3 B + 2/3 fit
            exactfit = spline_filter( griddata )  # see Unser
            griddata += prefilter * (exactfit - griddata)
            prefilter = False
        self.griddata = griddata
        self.prefilter = (prefilter == True)

        self.nmap = 0
        if maps != []:  # sanity check maps --
            assert len(maps) == dim, "maps must have len %d, not %d" % (
                dim, len(maps))
            for j, (map, n, l, h) in enumerate( zip( maps, griddata.shape, lo, hi )):
                if map is None  or  callable(map):
                    lo[j], hi[j] = 0, 1
                    continue
                maps[j] = map = np.asanyarray(map)
                self.nmap += 1
                assert len(map) == n, "maps[%d] must have len %d, not %d" % (
                    j, n, len(map) )
                mlo, mhi = map.min(), map.max()
                if not (l <= mlo <= mhi <= h):
                    print "Warning: Intergrid maps[%d] min %.3g max %.3g " \
                        "are outside lo %.3g hi %.3g" % (
                        j, mlo, mhi, l, h )
        self.maps = maps
        self._lo = lo  # caller's lo for linear, 0 nonlinear maps
        shape_1 = np.array( self.griddata.shape, float ) - 1
        self._linearmap = shape_1 / np.where( hi > lo, hi - lo, np.inf )  # 25jun

#...............................................................................
    def __call__( self, X, out=None ):
        """ query_values = Intergrid(...) ( query_points npt x dim )
        """
        X = np.asanyarray(X)
        assert X.shape[-1] == self.dim, ("the query array must have %d columns, "
                "but its shape is %s" % (self.dim, X.shape) )
        Xdim = X.ndim
        if Xdim == 1:
            X = np.asarray([X])  # in a single point -> out scalar
        if self.copy:
            X = X.copy()
        assert X.ndim == 2, X.shape
        npt = X.shape[0]
        if out is None:
            out = np.empty( npt, dtype=self.griddata.dtype )
        t0 = time()
        np.clip( X, self.loclip, self.hiclip, out=X )  # inplace
            # X nonlinear maps inplace --
        for j, map in enumerate(self.maps):
            if map is None:
                continue
            if callable(map):
                X[:,j] = map( X[:,j] )  # clip again ?
            else:
                # PWL e.g. [50 52 62 63] -> [0 1 2 3] --
                X[:,j] = np.interp( X[:,j], map, np.arange(len(map)) )
            # linear map the rest, inplace (affine_transform ?)
        if self.nmap < self.dim:
            X -= self._lo
            X *= self._linearmap  # (griddata.shape - 1) / (hi - lo)
        ## print "test X:", X[0]

#...............................................................................
        map_coordinates( self.griddata, X.T,
            order=self.order, prefilter=self.prefilter,
            mode="nearest",  # outside -> edge
                # test: mode="constant", cval=np.NaN,
            output=out )
        if self.verbose:
            print "Intergrid: %.3g msec  %d points in a %s grid  %d maps  order %d" % (
                (time() - t0) * 1000, npt, self.griddata.shape, self.nmap, self.order )
        return out if Xdim == 2  else out[0]

    at = __call__

# end intergrid.py