## @ingroup Methods-Geometry-Three_Dimensional
# orientation_product.py
# 
# Created:  Dec 2013, SUAVE Team
# Modified: Jan 2016, E. Botero
#           Jan 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import jax.numpy as jnp
from jax import lax
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
    
    assert T.ndim == 3
    
    if Bb.ndim == 3:
        C = np.einsum('aij,ajk->aik', T, Bb)
        
    elif Bb.ndim == 2:
        C  =  np.einsum('aij,aj->ai', T, Bb)

    else:
        raise Exception('bad B rank')
        
    return C


## @ingroup Methods-Geometry-Three_Dimensional
def orientation_product_jax(T,Bb):
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
    
    assert T.ndim == 3
    
    if Bb.ndim == 3:
        C = jnp.einsum('aij,ajk->aik', T, Bb,precision=lax.Precision.HIGHEST,optimize='False')
  
        
    elif Bb.ndim == 2:
        C = jnp.einsum('aij,aj->ai', T, Bb,precision=lax.Precision.HIGHEST,optimize='False')
           
  
    else:
        raise Exception('bad B rank')
        
    return C