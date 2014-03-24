""" evaluate_curve.py: Evaluate a NURBS curve """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Structure import Data
from SUAVE.Geometry.Three_Dimensional.NURBS.Attributes.Curve import Curve
from SUAVE.Geometry.Three_Dimensional.NURBS.Attributes.Surface import Surface

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def evaluate_curve(curve,u):

    """  NURBS Support function: compute the basis function N_i,p
    
         Inputs:    curve = NURBS curve class instance
                    u = parametric coordinate (float)

         Outputs:   coordinates on curve at u (numpy float array)

    """
    # find span
    span = FindSpan(curve,u)

    # compute basis functions
    N = BasisFunctions(curve,u,span)
    
    # compute coordinates
    C = np.zeros(curve.dims);
    W = 0.0
    for i in range(0,curve.p+1):                        # for (i = 0; i <= p; i++)
        C[0] += N[i]*curve.CPs.x[span-curve.p+i]*curve.w[span-curve.p+i]
        C[1] += N[i]*curve.CPs.y[span-curve.p+i]*curve.w[span-curve.p+i]
        if curve.dims == 3:
            C[2] += N[i]*curve.CPs.z[span-curve.p+i]*curve.w[span-curve.p+i]
        W += N[i]*curve.w[span-curve.p+i]

    return C/W
