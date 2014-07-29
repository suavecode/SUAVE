# cm_alphadot.py
# 
# Created:  Andrew Wendorff, June 2014
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Structure import (
    Data, Container, Data_Exception, Data_Warning,
)
import numpy as np

# ----------------------------------------------------------------------
#   Method
# ----------------------------------------------------------------------

def cm_alphadot(cm_i, ep_alpha, l_t, mac):
    """ output = SUAVE.Methods.Flight_Dynamics.Dynamic_Stablity.Full_Linearized_Equations.Supporting_Functions.cm_alphadot(cm_i, ep_alpha, l_t, mac) 
        Calculating the pitching moment coefficient with respect to the rate of change of the alpha of attack of the aircraft        
        Inputs:
                 
        Outputs:
                
        Assumptions:
        
        Source:
            J.H. Blakelock, "Automatic Control of Aircraft and Missiles" Wiley & Sons, Inc. New York, 1991, (Need page number)
    """

    # Generating Stability derivative

    cm_alphadot  = 2. * cm_i * ep_alpha * l_t / mac
    
    return cm_alphadot 