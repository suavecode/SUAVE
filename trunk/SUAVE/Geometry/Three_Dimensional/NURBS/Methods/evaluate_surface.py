""" evaluate_surface.py: Evaluate a NURBS surface """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Core import Data
from SUAVE.Geometry.Three_Dimensional.NURBS.Attributes.Curve import Curve
from SUAVE.Geometry.Three_Dimensional.NURBS.Attributes.Surface import Surface

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def evaluate_surface(surface,u,v):

    """  EvaluateSurface(surface,u,v): evaluate a surface at (u,v)
    
         Inputs:    surface = NURBSSurface class instance   (required)      
                    u = parametric coordinate 1             (required)      (float)
                    v = parametric coordinate 2             (required)      (float)

         Outputs:   coordinates on surface at (u,v)     (list of floats)

    """

    # package data
    uCurve = Curve()
    uCurve.p = surface.p
    uCurve.knots = surface.uknots

    vCurve = Curve()
    vCurve.p = surface.q
    vCurve.knots = surface.vknots

    # find patch
    uspan = FindSpan(uCurve,u)
    vspan = FindSpan(vCurve,v)

    # compute basis functions
    Nu = BasisFunctions(uCurve,u,uspan)
    Nv = BasisFunctions(vCurve,v,vspan)
    
    # compute coordinates
    S = np.zeros(3); W = 0.0
    for l in range(0,surface.q+1):                       # for (l = 0; l <= q; l++)
        t = np.zeros(3); w = 0.0
        vi = vspan - surface.q + l   
        for k in range(0,surface.p+1):                   # for (k = 0; k <= p; k++)
            ui = uspan - uCurve.p + k
            t[0] += Nu[k]*surface.CPs.x[ui][vi]*surface.w[ui][vi]
            t[1] += Nu[k]*surface.CPs.y[ui][vi]*surface.w[ui][vi]
            t[2] += Nu[k]*surface.CPs.z[ui][vi]*surface.w[ui][vi]
            w += Nu[k]*surface.w[ui][vi]

        S += Nv[l]*t
        W += Nv[l]*w

    return S/W
