## @ingroup Methods-Geometry-Three_Dimensional
# orientation_product.py
# 
# Created:  Dec 2013, SUAVE Team
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Orientation Product
# ----------------------------------------------------------------------

## @ingroup Methods-Geometry-Three_Dimensional
def orientation_product(T,Bb):
    """Computes the product of a tensor and a vector.

    Assumptions:
    None

    Source:
    N/A

    Inputs:
    T         [-] 3-dimensional array with rotation matrix
                  patterned along dimension zero
    Bb        [-] 3-dimensional vector

    Outputs:
    C         [-] transformed vector

    Properties Used:
    N/A
    """            
    
    assert np.rank(T) == 3
    
    if np.rank(Bb) == 3:
        C = np.einsum('aij,ajk->aik', T, Bb )
    elif np.rank(Bb) == 2:
        C = np.einsum('aij,aj->ai', T, Bb )
    else:
        raise Exception('bad B rank')
        
    return C