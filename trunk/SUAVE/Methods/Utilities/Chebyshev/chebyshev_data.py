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

def chebyshev_data(N,integration=False):

    N = int(N)

    # error checking:
    if N <= 0:
        print "N must be > 0"
        return []   

    D = np.zeros((N,N));

    # x array
    x = 0.5*(1 - np.cos(np.pi*np.arange(0,N)/(N-1)))

    # D operator
    c = np.array(2)
    c = np.append(c,np.ones(N-2))
    c = np.append(c,2)
    c = c*((-1)**np.arange(0,N));
    A = np.tile(x,(N,1)).transpose(); 
    dA = A - A.transpose() + np.eye(N); 
    cinv = 1/c; 

    for i in range(N):
        for j in range(N):
            D[i][j] = c[i]*cinv[j]/dA[i][j]

    D = D - np.diag(np.sum(D.transpose(),axis=0));

    # I operator
    if integration:
        I = np.linalg.inv(D[1:,1:]); 
        I = np.append(np.zeros((1,N-1)),I,axis=0)
        I = np.append(np.zeros((N,1)),I,axis=1)
        return x, D, I
    else:
        return x, D