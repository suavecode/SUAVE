## @ingroup Methods-Utilities
#soft_max.py
#Created:  Feb 2016, M. Vegh

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np


# ----------------------------------------------------------------------
# soft_max Method
# ----------------------------------------------------------------------
## @ingroup Methods-Utilities
def soft_max(x1,x2):
    """Computes the soft maximum of two inputs.

    Assumptions:
    None

    Source:
    http://www.johndcook.com/blog/2010/01/20/how-to-compute-the-soft-maximum/

    Inputs:
    x1   [-]
    x2   [-]

    Outputs:             
    f    [-] The soft max

    Properties Used:
    N/A
    """    
    max=np.maximum(x1,x2)
    min=np.minimum(x1,x2)
    f=max+np.log(1+np.exp(min-max))
    
    return f