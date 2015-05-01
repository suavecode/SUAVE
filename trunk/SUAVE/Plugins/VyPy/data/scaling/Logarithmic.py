
from ScalingFunction import ScalingFunction

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import numpy as np


# ----------------------------------------------------------------------
#   Logarithmic Scaling Function
# ----------------------------------------------------------------------

class Logarithmic(ScalingFunction):
    
    def __init__(self,scale=1.0,base=10.0):
        """ o / scl ==> np.log_base(other/scale)
            o * scl ==> (base**other) * scale
            
            base defualt to 10.0
            base could be numpy.e for example
        """
        self.scale = scale
        self.base   = base
        
    def set_scaling(self,other):
        return np.log10(other/self.scale)/np.log10(self.base)
    def unset_scaling(self,other):
        return (self.base**other)*self.scale
    
    
# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
    
if __name__ == '__main__':    
    
    import numpy as np
    
    s = Logarithmic(2.0,10.0)
    
    a = 20.0
    b = np.array([20,200,6000.])
    
    print a
    print b
    
    a = a / s    
    b = b / s
    
    print a
    print b
    
    a = a * s    
    b = b * s    
    
    print a
    print b