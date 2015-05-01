
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import copy

from VyPy.data import OrderedBunch
from ScalingFunction import ScalingFunction


# ----------------------------------------------------------------------
#   Linear Scaling Function
# ----------------------------------------------------------------------

class ScalingBunch(OrderedBunch,ScalingFunction):
    
    def set_scaling(self,Other):
        
        Other = copy.deepcopy(Other)
        
        for key in Other.keys():
            if self.has_key(key):
                Other[key] = Other[key] / self[key]
        
        return Other
    
    def unset_scaling(self,Other):
        
        Other = copy.deepcopy(Other)
        
        for key in Other.keys():
            if self.has_key(key):
                Other[key] = Other[key] * self[key]
        
        return Other               

    
# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------

if __name__ == '__main__':
    
    import numpy as np
    
    from Linear import Linear
    
    S = ScalingBunch()
    S.X = Linear(10.0,0.0)
    S.Y = Linear(2.0,1.0)
    
    data = OrderedBunch()
    data.X = 10.0
    data.Y = np.array([1,2,3.])
    
    print data
    
    data = data / S
    
    print data
    
    data = data * S
    
    print data
    
    