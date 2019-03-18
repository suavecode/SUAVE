## @ingroup Methods-Utilities-Chebyshev
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Method
# ----------------------------------------------------------------------

## @ingroup Methods-Utilities-Chebyshev
def linear_data(N = 16, integration = True, **options):
    """Calculates the differentiation and integration matricies
    using chebyshev's pseudospectral algorithm, based on linearly
    spaced samples in x.
    
    D and I are not symmetric
    get derivatives with df_dy = np.dot(D,f)
    get integral with    int_f = np.dot(I,f)
        where f is either a 1-d vector or 2-d column array
        
    A full example of how these operators are used is available in 
    the chebyshev_data.py (same folder)

    Assumptions:
    None

    Source:
    N/A

    Inputs:
    N                      [-]        Number of points
    integration (optional) <boolean>  Determines if the integration operator is calculated

    Outputs:
    x                      [-]        N-number of cosine spaced control points, in range [0,1]
    D                      [-]        Differentiation operation matrix
    I                      [-]        Integration operation matrix, or None if integration = False

    Properties Used:
    N/A
    """           
    
    # setup
    N = int(N)
    if N <= 0: raise RuntimeError("N = %i, must be > 0" % N)
    
    
    # --- X vector
    
    # linear spaced in range [0,1]
    x = np.linspace(0,1,N)   


    # --- Differentiation Operator
    
    # coefficients
    c = np.array( [2.] + [1.]*(N-2) + [2.] )
    c = c * ( (-1.) ** np.arange(0,N) )
    A = np.tile( x, (N,1) ).T
    dA = A - A.T + np.eye( N )
    cinv = 1./c; 

    # build operator
    D = np.zeros( (N,N) );
    
    # math
    c    = np.array(c)
    cinv = np.array([cinv])
    cs   = np.multiply(c,cinv.T)
    D    = np.divide(cs.T,dA)

    # more math
    D = D - np.diag( np.sum( D.T, axis=0 ) );

    # --- Integration operator
    
    if integration:
        # invert D except first row and column
        I = np.linalg.inv(D[1:,1:]); 
        
        # repack missing columns with zeros
        I = np.append(np.zeros((1,N-1)),I,axis=0)
        I = np.append(np.zeros((N,1)),I,axis=1)
        
    else:
        I = None
        
    # done!
    return x, D, I