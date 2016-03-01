# ep_alpha.py
# 
# Created:  Jun 2014, A. Wendorff
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#   Method
# ----------------------------------------------------------------------

def ep_alpha(cL_w_alpha, Sref, span):
    """ output = SUAVE.Methods.Flight_Dynamics.Dynamic_Stablity.Full_Linearized_Equations.Supporting_Functions.ep_alpha(cL_w_alpha, Sref, span, e) 
        Calculating the change in the downwash with change in angle of attack         
        Inputs:
                 
        Outputs:
                
        Assumptions:
        
        Source:
            J.H. Blakelock, "Automatic Control of Aircraft and Missiles" Wiley & Sons, Inc. New York, 1991, (34)
    """

    # Generating Stability derivative
    ep_alpha = 2 * cL_w_alpha/ np.pi / (span ** 2. / Sref )
    
    return ep_alpha