# cx_u.py
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

def cx_u(cD):
    """ output = SUAVE.Methods.Flight_Dynamics.Dynamic_Stablity.Full_Linearized_Equations.Supporting_Functions.cx_u(cD) 
        Calculating the coefficient of force in the x direction with respect to the change in forward velocity of the aircraft        
        Inputs:
                 
        Outputs:
                
        Assumptions:
        
        Source:
            J.H. Blakelock, "Automatic Control of Aircraft and Missiles" Wiley & Sons, Inc. New York, 1991, (Need page number)
    """

    # Generating Stability derivative

    cx_u  = -2. * cD
    
    return cx_u