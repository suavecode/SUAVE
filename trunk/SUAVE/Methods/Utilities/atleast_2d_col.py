import numpy as np

def atleast_2d_col(A,oned_as='col'):
    ''' B = atleast_2d_col(A,oned_as='col')
        ensures A is an array and at least of rank 2
        defaults 1d arrays to column vectors
        can override with oned_as = 'row'
        
        Note: B contains a pointer to A, so modifications of A
        will affect B. Except if A was a scalar.
    '''
    if not isinstance(A,np.ndarray):
        A = np.array([A])
    if np.rank(A) < 2:
        if oned_as == 'row':
            A = A[None,:]
        elif oned_as == 'col':
            A = A[:,None]
        else:
            raise Exception , "oned_as must be 'row' or 'col' "
            
    return A