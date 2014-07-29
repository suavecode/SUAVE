# longitudinal.py
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

def cm_q(cm_i, l_t, mac):
    """ output = SUAVE.Methods.Flight_Dynamics.Dynamic_Stablity.Full_Linearized_Equations.Supporting_Functions.cz_u(cL, U, cL_u = 0) 
        Calculating the pitching moment coefficient with respect to the perturbational angular rate in the y-body-axis         
        Inputs:
        
        Outputs:
        
        Assumptions:
        
        Source:
            J.H. Blakelock, "Automatic Control of Aircraft and Missiles" Wiley & Sons, Inc. New York, 1991, (Need page number)
    """

    # Generating Stability derivative
    
    cm_q = 2. * 1.1 * cm_i * l_t / mac # Check function
    
    return cm_q