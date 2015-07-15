# cz_alpha.py
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

def cz_alpha(cD, cL_alpha):
    """ output = SUAVE.Methods.Flight_Dynamics.Dynamic_Stablity.Full_Linearized_Equations.Supporting_Functions.cz_alpha(cD, cL_alpha) 
        Calculating the coefficient of force in the z-direction with respect to alpha of attack of the aircraft        
        Inputs:
                 
        Outputs:
                
        Assumptions:
        
        Source:
            J.H. Blakelock, "Automatic Control of Aircraft and Missiles" Wiley & Sons, Inc. New York, 1991, (Need page number)
    """

    # Generating Stability derivative

    cz_alpha  = -cD - cL_alpha
    
    return cz_alpha 