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


def cosine_space(N,x0,xf):

    N = int(N)

    # error checking:
    if N <= 0:
        print "N must be > 0"
        return []   

    # x array
    x = x0 + 0.5*(1 - np.cos(np.pi*np.arange(0,N)/(N-1)))*(xf - x0)

    return x
