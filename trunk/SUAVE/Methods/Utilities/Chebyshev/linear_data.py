# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import autograd.numpy as np 

# ----------------------------------------------------------------------
#  Method
# ----------------------------------------------------------------------

def linear_data(N = 16, integration = True, **options):
    """ x, D, I = linear_data(N,integration=True)
        calcualtes differentiation and integration matricies
        using chebychev's pseudospectral algorithm, based on
        cosine spaced samples in x.
        
        Inputs:
            N - number of control points, 
                default 16 is quite accurate for most cases
            integration - optional, if false, skips the calculation of I, 
                and returns None
        
        Outputs:
            x - N-number of cosine spaced control points, in range [0,1]
            D - differentiation operation matrix
            I - integration operation matrix, or None if integration=False
            
        Usage Notes - 
            D and I are not symmetric
            get derivatives with df_dy = np.dot(D,f)
            get integral with    int_f = np.dot(I,f)
                where f is either a 1-d vector or 2-d column array
        
        Example:
            How to calculate derivatives and integrals
            
            # get the data
            x,D,I = linear_data(16)
            
            # the function
            def func(x):
                return x ** 2. + 1.
            
            # scaling and offsets from nondimensional x to dimensional y
            dy_dx = 10. 
            y0    = -4.
            
            # scale to dimensional
            y = x * dy_dx + y0
            D = D / dy_dx # yup, divide
            I = I * dy_dx
            
            # the function
            f = func(y)  
            
            # the derivative and integrals
            df_dy = np.dot(D,f)
            int_f = np.dot(I,f)
            
            # plot
            import pylab as plt
            plt.subplot(3,1,1)
            plt.plot(y,f)
            plt.ylabel('f(y)')
            plt.subplot(3,1,2)
            plt.plot(y,df_dy)
            plt.ylabel('df/dy')
            plt.subplot(3,1,3)
            plt.plot(y,int_f)    
            plt.ylabel('int(f(y))')
            plt.xlabel('y')
            plt.show()
            
    """
    
    # setup
    N = int(N)
    if N <= 0: raise RuntimeError , "N = %i, must be > 0" % N
    
    
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