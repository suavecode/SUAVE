# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Method
# ----------------------------------------------------------------------

def chebyshev_data(N,integration=True):
    """ x, D, I = chebyshev_data(N,integration=True)
        calcualtes differentiation and integration matricies
        using chebychev's pseudospectral algorithm, based on
        cosine spaced samples in x.
        
        Inputs:
            N - number of control points
            integration (optional) - if false, skips the calculation of I, 
                                     and returns None
        
        Outputs:
            x - N-number of cosine spaced control points, in range [0,1]
            D - differentiation operation matrix
            I - integration operation matrix, or None if integration=False
        
        Example:
            How to calculate derivatives and integrals
            
            # scaling factor from nondimensional x to dimensional y
            dy_dx = 10. 
            
            # scale to dimensional
            y = x * dy_dx
            D = D / dy_dx # yup, divide
            I = I * dy_dx
            
            # the function
            f = func(y)  # np.array([1.,2.,3.,...])
            
            # the derivative and integrals
            df_dy = np.dot(D,f)
            int_f = np.dot(I,f)
            
            # plot
            plt.plot(y,f)
            plt.plot(y,df_dy)
            plt.plot(y,int_f)
            
    """
    
    # setup
    N = int(N)
    if N <= 0: raise RuntimeError , "N = %i, must be > 0" % N
    
    
    # --- X vector
    
    # cosine spaced in range [0,1]
    x = 0.5*(1 - np.cos(np.pi*np.arange(0,N)/(N-1)))    


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
    for i in range(N):
        for j in range(N):
            D[i][j] = c[i]*cinv[j]/dA[i][j]

    # more math
    D = D - np.diag( np.sum( D.T, axis=0 ) );


    # --- Integratin operator
    
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