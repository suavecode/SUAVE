# wave_drag_volume.py
# 
# Created:  Tim MacDonald, 6/24/14
# Modified: Tim MacDonald, 6/24/14
# Based on http://adg.stanford.edu/aa241/drag/ssdragcalc.html

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Attributes.Gases import Air # you should let the user pass this as input
from SUAVE.Attributes.Results.Result import Result
air = Air()
compute_speed_of_sound = air.compute_speed_of_sound

# python imports
import os, sys, shutil
from copy import deepcopy
from warnings import warn

# package imports
import numpy as np
import scipy as sp


# ----------------------------------------------------------------------
#   The Function
# ----------------------------------------------------------------------



def wave_drag_volume(conditions,configuration,wing):
    """ SUAVE.Methods.wave_drag_lift(conditions,configuration,fuselage)
        computes the wave drag due to lift 
        
        Inputs:

        Outputs:

        Assumptions:

        
    """

    # unpack inputs
    freestream = conditions.freestream
    
    total_length = wing.chord_root
    Sref = wing.sref
    
    # conditions
    Mc  = freestream.mach_number

    # length-wise aspect ratio
    ARL = total_length**2/Sref
    #print "ARL = " + str(ARL)
    
    # thickness to chord
    t_c_w = wing.t_c
    
    # Computations
    x = np.pi*ARL/4
    beta = np.sqrt(Mc**2-1)
    wave_drag_volume = 4*t_c_w**2*(beta**2+2*x**2)/(beta**2+x**2)**1.5

    
    # dump data to conditions
    #wave_volume_result = Result(
        #reference_area            = Sref   , 
        #wave_drag_lift_coefficient = wave_drag_lift ,
        #length_AR                 = ARL,
    #)
    #conditions.aerodynamics.drag_breakdown.parasite[fuselage.tag] = fuselage_result    
    # Create output here
    
    return wave_drag_volume*1.15


# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__': 
    
    #t_c_w = 0.03
    #Mc = 2.4
    #ARL = 4
    # Result = 0.00158 According to http://adg.stanford.edu/aa241/drag/ssdragcalc.html
    
    ## Computations
    #x = np.pi*ARL/4
    #beta = np.sqrt(Mc**2-1)
    #wave_drag_volume = 4*t_c_w**2*(beta**2+2*x**2)/(beta**2+x**2)**1.5  
    
    raise NotImplementedError
    #return wave_drag_lift