## @ingroup Methods-Aerodynamics-Fidelity_Zero-Lift
# vortex_lift.py
# 
# Created:  Jub 2014, T. MacDonald
# Modified: Jul 2014, T. MacDonald
#           Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#   Vortex Lift
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Fidelity_Zero-Lift
def vortex_lift(AoA,configuration,wing):
    """Computes vortex lift

    Assumptions:
    wing capable of vortex lift

    Source:
    adg.stanford.edu (Stanford AA241 A/B Course Notes)
    
    Inputs:
    wing.aspect_ratio         [Unitless]
    wing.sweeps.quarter_chord [radians]

    Outputs:
    CL_prime  (added CL)      [Unitless]

    Properties Used:
    N/A
    """  

    AR    = wing.aspect_ratio
    GAMMA = wing.sweeps.quarter_chord
    
    # angle of attack
    a = AoA
    
    # lift coefficient addition
    CL_prime = np.pi*AR/2*np.sin(a)*np.cos(a)*(np.cos(a)+np.sin(a)*np.cos(a)/np.cos(GAMMA)-np.sin(a)/(2*np.cos(GAMMA)))

    return CL_prime