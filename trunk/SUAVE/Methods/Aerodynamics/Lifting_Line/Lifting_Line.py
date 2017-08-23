## @ingroup Methods-Aerodynamics-Fidelity_Zero-Lift
# wing_compressibility_correction.py
# 
# Created:  Aug 2017, E. Botero
# Modified: 
#           

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  The Function
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Fidelity_Zero-Lift
def lifting_line(state,settings,geometry):
    """

    Assumptions:
    subsonic and unswept

    Source:
    Traub, L. W., Botero, E., Waghela, R., Callahan, R., & Watson, A. (2015). Effect of Taper Ratio at Low Reynolds Number. Journal of Aircraft.
    
    Inputs:
    N/A

    Outputs:
    N/A

    Properties Used:
    N/A
    """  
    
    
    # unpack
    b   = None # Wingspan
    rho = None # Freestream density
    V   = None # Freestream velocity
    mu  = None # Freestream viscosity
    r   = 20   # Need to set somewhere
    
    N      = r-1                  # number of spanwise divisions
    n      = np.linspace(0,N,N)   # vectorize
    thetan = n*np.pi/r            # angular stations
    yn     = -b*np.cos(thetan)/2. # y locations based on the angular spacing
    etan   = np.abs(2.*yn/b)       # normalized coordinates
    
    # Need to project the spanwise y locations into the 
    
   
    pass