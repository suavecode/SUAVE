""" basis_function.py: NURBS basis functions """

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

def basis_function(p,m,knots,i,u):

    """  NURBS Support function: compute the basis function N_i,p
    
         Inputs:    p = degree of basis functions (int)
                    m = number of knot spans (m + 1 knots) (int)
                    knots = knot vector (length m) (floats)
                    i = knot span (int)
                    u = parametric coordinate (float)

         Outputs:   N_i,p (float)

    """
    i = int(i); p = int(p); m = int(m)
    N = np.ones(p)

    # special cases
    if i == 0 and u == knots[0]:
        return 1.0
    if i == (m - p - 1) and u == knots[m]:
        return 1.0

    # locality
    if u < knots[i] or u >= knots[i+p+1]:
        return 0.0

    # initialize 0th degree functions
    for j in range(1,p):

        if u >= knots[i+j] and u < knots[i+j+1]:
            N[j] = 1.0
        else:
            N[j] = 0.0

    for k in range(1,p):

        if N[0] == 0.0:
            s = 0.0
        else:
            s = ((u - knots[i])*N[0])/(knots[i+k] - knots[i])
        
        for j in range(0,p-k+1):
            left = knots[i+j+1]
            right = knots[i+j+k+1]
            if N[j+1] == 0.0:
                N[j] = s
                s = 0.0
            else:
                t = N[j+1]/(right - left)
                N[j] = s + (right - u)*t
                s = (u - left)*t
     
    return N[0]