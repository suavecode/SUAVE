# orientation_product.py
# 
# Created:  Dec 2013, SUAVE Team
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Orientation Transpose
# ----------------------------------------------------------------------

def orientation_transpose(T):
    
    assert np.rank(T) == 3
    
    Tt = np.swapaxes(T,1,2)
        
    return Tt