
import numpy as np

def orientation_product(T,Bb):
    
    assert np.rank(T) == 3
    
    if np.rank(Bb) == 3:
        C = np.einsum('aij,ajk->aik', T, Bb )
    elif np.rank(Bb) == 2:
        C = np.einsum('aij,aj->ai', T, Bb )
    else:
        raise Exception , 'bad B rank'
        
    return C