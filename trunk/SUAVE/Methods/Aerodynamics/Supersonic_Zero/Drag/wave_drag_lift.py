# wave_drag_lift.py
# 
# Created:  Tim MacDonald, 6/24/14
# Modified: Tim MacDonald, 7/14/14
# Based on http://adg.stanford.edu/aa241/drag/ssdragcalc.html

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Attributes.Gases import Air # you should let the user pass this as input
from SUAVE.Core import Results
air = Air()
compute_speed_of_sound = air.compute_speed_of_sound

# python imports
import os, sys, shutil
from copy import deepcopy
import copy
from warnings import warn

# package imports
import numpy as np
import scipy as sp


# ----------------------------------------------------------------------
#   The Function
# ----------------------------------------------------------------------



def wave_drag_lift(conditions,configuration,wing):
    """ SUAVE.Methods.wave_drag_lift(conditions,configuration,wing)
        computes the wave drag due to lift 
        
        Inputs:
        - SUave wing
        - Sref - wing reference area
        - Mc - mach number
        - CL - coefficient of lift
        - total_length - length of the wing root

        Outputs:
        - CD due to wave drag from the wing

        Assumptions:
        - Supersonic mach numbers
        - Reference area of passed wing is desired for CD
        
    """

    # Unpack
    freestream = conditions.freestream
    total_length = wing.chords.root
    Sref = wing.areas.reference
    
    # Conditions
    Mc  = copy.copy(freestream.mach_number)

    # Length-wise aspect ratio
    ARL = total_length**2/Sref
    
    # Lift coefficient
    CL = copy.copy(conditions.aerodynamics.lift_coefficient)
    
    # Computations
    x = np.pi*ARL/4
    beta = np.array([[0.0]] * len(Mc))
    beta[Mc >= 1.05] = np.sqrt(Mc[Mc >= 1.05]**2-1)
    wave_drag_lift = np.array([[0.0]] * len(Mc))
    wave_drag_lift[Mc >= 1.05] = CL[Mc >= 1.05]**2*x/4*(np.sqrt(1+(beta[Mc >= 1.05]/x)**2)-1)
    wave_drag_lift[0:len(Mc[Mc >= 1.05]),0] = wave_drag_lift[Mc >= 1.05]

    
    # Dump data to conditions
    wave_lift_result = Results(
        reference_area            = Sref   , 
        wave_drag_lift_coefficient = wave_drag_lift ,
        length_AR                 = ARL,
    )

    return wave_drag_lift


# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__': 
    
    #CL = 0.1
    #Mc = 2.4
    #ARL = 4
    # Result = 0.00171 According to http://adg.stanford.edu/aa241/drag/ssdragcalc.html
    
    ## Computations
    #x = np.pi*ARL/4
    #beta = np.sqrt(Mc**2-1)
    #wave_drag_lift = CL**2*x/4*(np.sqrt(1+(beta/x)**2)-1)    
    
    raise NotImplementedError
    #return wave_drag_lift