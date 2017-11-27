## @ingroup Methods-Flight_Dynamics-Dynamic_Stability-Full_Linearized_Equations-Supporting_Functions
# cy_psi.py
# 
# Created:  Jun 2014, A. Wendorff
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

## @ingroup Methods-Flight_Dynamics-Dynamic_Stability-Full_Linearized_Equations-Supporting_Functions
import numpy as np

# ----------------------------------------------------------------------
#   Method
# ----------------------------------------------------------------------

def cy_psi(cL,theta):
    """ This calculates the force coefficient in the y direction 
    with respect to the yaw angle of the aircraft        

    Assumptions:
    None

    Source:
    J.H. Blakelock, "Automatic Control of Aircraft and Missiles" 
    Wiley & Sons, Inc. New York, 1991, (pg 23)

    Inputs:
    theta                [radians]
    cL                   [dimensionless]

    Outputs:
    cy_psi               [dimensionless]

    Properties Used:
    N/A           
    """

    # Generating Stability derivative
    cy_psi = cL * np.tan(theta)
    
    return cy_psi