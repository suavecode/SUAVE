## @ingroup Methods-Utilities
# Cubic_Spline_Blender.py
# 
# Created:  Feb 2019, T. MacDonald
# Modified: Jan 2020, T. MacDonald (moved from Method/Aerodynamics/Supersonic_Zero/Drag)

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Blender Class
# ----------------------------------------------------------------------

## @ingroup Methods-Utilities
class Cubic_Spline_Blender():
    """This is a cubic spline function that can be used to blend two type of calculations
    without knowing the end points. It preserves continuous first derivatives.

    Assumptions:
    None

    Source:
    Information at:
    https://en.wikipedia.org/wiki/Cubic_Hermite_spline
    """ 
    
    def __init__(self, x_start, x_end):
        """This sets the default start and end position.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        N/A
        """           
        self.x_start = x_start
        self.x_end   = x_end
    
    
    def compute(self,x):
        """This computes the y value along a normalized spline

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        N/A
        """          
        eta = self.eta_transform(x)
    
        y = 2*eta*eta*eta-3*eta*eta+1
        y[eta<0] = 1
        y[eta>1] = 0
        return y

    def eta_transform(self,x):
        """Normalizes the transformation

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        N/A
        """          
        x_start = self.x_start
        x_end   = self.x_end
        
        eta     = (x-x_start)/(x_end-x_start)
        
        return eta