## @ingroup Methods-Geometry-Three_Dimensional
# orientation_product.py
# 
# Created:  Dec 2013, SUAVE Team
# Modified: Jan 2016, E. Botero
#           Jan 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Orientation Transpose
# ----------------------------------------------------------------------

## @ingroup Methods-Geometry-Three_Dimensional
def orientation_transpose(T):
    """Computes the transpose of a tensor.

    Assumptions:
    None

    Source:
    N/A

    Inputs:
    T         [-] 3-dimensional array with rotation matrix
                  patterned along dimension zero

    Outputs:
    Tt        [-] transformed tensor

    Properties Used:
    N/A
    """   
    
    assert T.ndim == 3
    
    Tt = np.swapaxes(T,1,2)
        
    return Tt