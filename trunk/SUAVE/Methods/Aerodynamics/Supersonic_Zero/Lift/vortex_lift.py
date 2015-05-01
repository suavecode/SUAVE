# vortex_lift.py
# 
# Created:  Tim MacDonald, 6/27/14
# Modified: Tim MacDonald, 7/14/14
# Based on http://adg.stanford.edu/aa241/highlift/sstclmax.html

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Attributes.Gases import Air # you should let the user pass this as input
from SUAVE.Core import Results

# python imports
import os, sys, shutil
from copy import deepcopy
from warnings import warn

# package imports
import numpy as np

# ----------------------------------------------------------------------
#   The Function
# ----------------------------------------------------------------------

#def vortex_lift(AoA,configuration,wing):
def vortex_lift(state,settings,geometry):
    """ SUAVE.Methods.wave_drag_lift(conditions,configuration,wing)
        computes the vortex lift on highly swept wings
        
        Inputs:
        - SUave wing and angles of attack

        Outputs:
        - CL due to vortex lift

        Assumptions:
        - Wing with high sweep

        
    """
    
    Mc        = state.conditions.freestream.mach_number
    AoA       = state.conditions.aerodynamics.angle_of_attack
    
    wings_lift = state.conditions.aerodynamics.lift_coefficient

    vortex_cl = np.array([[0.0]] * len(Mc))
    
    

    for wing in geometry.wings:
    
        if wing.vortex_lift is True:
            #vortex_cl[Mc < 1.0] = vortex_lift(AoA[Mc < 1.0],configuration,wing) # This was initialized at 0.0
            AR = wing.aspect_ratio
            GAMMA = wing.sweep
            
            # angle of attack
            a = AoA[Mc < 1.0]
            
            # lift coefficient addition
            vortex_cl[Mc < 1.0] += np.pi*AR/2*np.sin(a)*np.cos(a)*(np.cos(a)+np.sin(a)*np.cos(a)/np.cos(GAMMA)-np.sin(a)/(2*np.cos(GAMMA)))
    
    
    wings_lift[Mc <= 1.05] = wings_lift[Mc <= 1.05] + vortex_cl[Mc <= 1.05]    
    
    state.conditions.aerodynamics.lift_breakdown.vortex_lift = vortex_cl   
    
    return wings_lift