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

def vortex_lift(AoA,configuration,wing):
    """ SUAVE.Methods.wave_drag_lift(conditions,configuration,wing)
        computes the vortex lift on highly swept wings
        
        Based on http://adg.stanford.edu/aa241/highlift/sstclmax.html
        
        Inputs:
        - SUave wing and angles of attack

        Outputs:
        - CL due to vortex lift

        Assumptions:
        - Wing with high sweep

        
    """
    
    AR    = wing.aspect_ratio
    GAMMA = wing.sweeps.quarter_chord
    
    # angle of attack
    a = AoA
    
    # lift coefficient addition
    CL_prime = np.pi*AR/2*np.sin(a)*np.cos(a)*(np.cos(a)+np.sin(a)*np.cos(a)/np.cos(GAMMA)-np.sin(a)/(2*np.cos(GAMMA)))
    
    
    return CL_prime