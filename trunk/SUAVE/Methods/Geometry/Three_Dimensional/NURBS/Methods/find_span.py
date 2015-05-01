""" find_span.py: find NURBS curve parametric span """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Core import Data
from SUAVE.Methods.Geometry.Three_Dimensional.NURBS.Attributes.Curve import Curve
from SUAVE.Methods.Geometry.Three_Dimensional.NURBS.Attributes.Surface import Surface

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def find_span(curve,u):

    """  NURBS Support function: determine the know span index
    
         Inputs:    curve = NURBS curve instance
                    u = parametric coordinate

         Outputs:   knot span index (int)

    """

    # unpack
    m = len(curve.knots) 
    p = curve.p
    n = m - p - 1

    # special cases of endpoint
    if u == curve.knots[-1]:
        return n - 1

    # binary search
    low = 0; high = m
    mid = (low + high)/2    # note: ints
    while u < curve.knots[mid] or u >= curve.knots[mid+1]:
        if u < curve.knots[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high)/2

    return mid