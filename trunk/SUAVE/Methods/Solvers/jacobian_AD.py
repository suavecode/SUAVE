""" jacobian_AD.py: compute Jacobian through automatic differentition """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Plugins.ADiPy import jacobian, ad
from residuals import residuals

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def jacobian_AD(x,problem):

    xi = ad(x,np.eye(len(x)))
    J = jacobian(residuals(xi,problem))

    return J