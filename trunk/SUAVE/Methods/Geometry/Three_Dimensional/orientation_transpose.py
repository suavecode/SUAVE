
import numpy as np

def orientation_transpose(T):
    
    assert np.rank(T) == 3
    
    Tt = np.swapaxes(T,1,2)
        
    return Tt