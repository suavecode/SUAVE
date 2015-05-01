# longitudinal.py
# 
# Created:  Andrew Wendorff, June 2014
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import (
    Data, Container, Data_Exception, Data_Warning,
)
import numpy as np

# ----------------------------------------------------------------------
#   Method
# ----------------------------------------------------------------------

def cn_p(cLw,depdalpha):
    """ output = SUAVE.Methods.Flight_Dynamics.Dynamic_Stablity.Full_Linearized_Equations.Supporting_Functions.cn_p(CLw,depdalpha) 
        Calculating the yawing moment coefficient with respect to perturbational angular rate around the x-body-axis        
        Inputs:
                 
        Outputs:
                
        Assumptions:
        
        Source:
            J.H. Blakelock, "Automatic Control of Aircraft and Missiles" Wiley & Sons, Inc. New York, 1991, (Need page number)
    """

    # Generating Stability derivative
    
    cn_p = -cLw/8. * (1. - depdalpha)
    
    return cn_p