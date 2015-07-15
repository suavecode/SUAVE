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

def vortex_lift(AoA,configuration,wing):
    """ SUAVE.Methods.wave_drag_lift(conditions,configuration,wing)
        computes the vortex lift on highly swept wings
        
        Inputs:
        - SUave wing and angles of attack

        Outputs:
        - CL due to vortex lift

        Assumptions:
        - Wing with high sweep

        
    """

    
    AR = wing.aspect_ratio
    GAMMA = wing.sweep
    
    # angle of attack
    a = AoA
    
    # lift coefficient addition
    CL_prime = np.pi*AR/2*np.sin(a)*np.cos(a)*(np.cos(a)+np.sin(a)*np.cos(a)/np.cos(GAMMA)-np.sin(a)/(2*np.cos(GAMMA)))
    
    
    return CL_prime