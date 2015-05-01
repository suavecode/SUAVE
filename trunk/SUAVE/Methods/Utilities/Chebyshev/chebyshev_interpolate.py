
""" Utilities.py: Mathematical tools and numerical integration methods for ODEs """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
#import ad
from SUAVE.Core import Data


# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def chebyshev_interpolate(x0,xf,c,x):

    N = len(c)

    # error checking:
    if N < 2:
        print "N must be > 1"
        return []   
    else:
        f = np.zeros(N)

    y = (2.0*x-x0-xf)/(xf-x0); y2 = 2.0*y
    d = 0.0; dd = 0.0
    for j in range(N-1,0,-1):
        sv = d
        d = y2*d - dd + c[j]
        dd = sv

    return y*d - dd + 0.5*c[0]
