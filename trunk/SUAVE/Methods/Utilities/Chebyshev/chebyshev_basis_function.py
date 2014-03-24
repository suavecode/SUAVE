""" Utilities.py: Mathematical tools and numerical integration methods for ODEs """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
#import ad
from scipy.optimize import root   #, fsolve, newton_krylov
from SUAVE.Structure import Data
from SUAVE.Attributes.Results import Segment

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------


def chebyshev_basis_function(n,x):

    n = int(n)

    # base cases
    if n == 0:
        return np.ones(len(x))
    elif n == 1:
        return x
    else:
        return 2*x*chebyshev_basis_function(n-1,x) - \
            chebyshev_basis_function(n-2,x)
