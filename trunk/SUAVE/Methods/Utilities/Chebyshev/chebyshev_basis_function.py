""" Utilities.py: Mathematical tools and numerical integration methods for ODEs """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Core import Data

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
