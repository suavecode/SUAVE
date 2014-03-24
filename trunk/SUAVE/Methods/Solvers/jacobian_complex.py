""" Utilities.py: Mathematical tools and numerical integration methods for ODEs """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from residuals import residuals

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def jacobian_complex(x,problem):

    # Jacobian via complex step 
    problem.complex = True
    h = 1e-12; N = len(x)
    J = np.zeros((N,N))

    for i in xrange(N):

        xi = x + 0j
        xi[i] = np.complex(x[i],h)
        R = residuals(xi,problem)
        J[:,i] = np.imag(R)/h

    problem.complex = False
    return J