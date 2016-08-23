""" basis_functions.py: NURBS basis functions """

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

def basis_functions(curve,u,i):

    """  NURBS Support function: compute all nonzero basis function values
    
         Inputs:    i = knot span
                    u = parametric coordinate
                    p = degree of basis functions
                    knots = knot vector (length m)

         Outputs:   length p+1 array of basis function values (float)

    """
    i = int(i); p = int(curve.p)
    N = np.ones(p+1); left = np.zeros(p+1); right = np.zeros(p+1)

    for j in range(1,p+1):              # for (j = 1; j <= p; j++)

        left[j] = u - curve.knots[i+1-j]
        right[j] = curve.knots[i+j] - u
        s = 0.0
        #print "j = " + str(j)

        for r in range(j):              # for (r = 0; r < j; r++)
            t = N[r]/(right[r+1] + left[j-r])
            N[r] = s + right[r+1]*t
            s = left[j-r]*t
            #print "r = " + str(r)
            
        N[j] = s
        # print u, N

    #raw_input()

    return N