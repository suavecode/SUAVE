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

def assign_values(A,B,irange,jrange):

    n, m = np.shape(B)
    if n != len(irange):
        print "Error: rows do not match between A and B"
        return
    if m != len(jrange):
        print "Error: columsn do not match between A and B"
        return

    ii = 0
    for i in irange:
        jj = 0
        for j in jrange:
            A[i][j] = B[ii][jj]
            jj += 1
        ii += 1

    return