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


def chebyshev_fit(x0,xf,f):

    N = len(f)

    # error checking:
    if N < 2:
        print "N must be > 1"
        return []   
    else:
        c = np.zeros(N)

    fac = 2.0/N
    for j in range(N):
        sum = 0.0
        for k in range(N):
            sum += f[N-1-k]*np.cos(np.pi*j*(k+0.5)/N)
        c[j] = fac*sum

    return c
