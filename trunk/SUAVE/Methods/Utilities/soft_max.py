#soft_max.py
#Created:  Feb 2016, M. Vegh

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import autograd.numpy as np 


# ----------------------------------------------------------------------
# soft_max Method
# ----------------------------------------------------------------------

def soft_max(x1,x2):

    """ f=soft_max(x1,x2)
        computes the soft_maximum of two inputs, so that it is differentiable
        uses the method from http://www.johndcook.com/blog/2010/01/20/how-to-compute-the-soft-maximum/
        to prevent potential overflow issues
         Inputs:    x1
                    x2

         Outputs:   f                                                 
        """
    max=np.maximum(x1,x2)
    min=np.minimum(x1,x2)
    f=max+np.log(1+np.exp(min-max))
    
    return f