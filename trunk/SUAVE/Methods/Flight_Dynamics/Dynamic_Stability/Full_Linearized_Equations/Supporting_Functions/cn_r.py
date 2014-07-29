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

def cn_r(cDw, Sv, Sref, l_v, span, eta_v, cLv_alpha):
    """ output = SUAVE.Methods.Flight_Dynamics.Dynamic_Stablity.Full_Linearized_Equations.Supporting_Functions(cDw, Sv, Sref, l_v, span, eta_v, cLv_alpha) 
        Calculating the yawing moment coefficient with respect to perturbational angular rate around the z-body-axis        
        Inputs:
        
        Outputs:
        
        Assumptions:
        
        Source:
            J.H. Blakelock, "Automatic Control of Aircraft and Missiles" Wiley & Sons, Inc. New York, 1991, (Need page number)
    """

    # Generating Stability derivative
    
    cn_r = -cDw/4. - 2. * Sv / Sref * (l_v/span)**2. * cLv_alpha * eta_v
    
    return cn_r