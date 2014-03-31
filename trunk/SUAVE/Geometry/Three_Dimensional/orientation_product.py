
import numpy as np

def orientation_product(A,B):
    
    assert np.rank(A) == 3
    
    if np.rank(B) == 3:
        C = np.einsum('aij,ajk->aik', A, B )
    elif np.rank(B) == 2:
        C = np.einsum('aij,aj->ai', A, B )
    else:
        raise Exception , 'bad B rank'
        
    return C