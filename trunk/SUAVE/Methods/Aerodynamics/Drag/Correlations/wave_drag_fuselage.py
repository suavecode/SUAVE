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



def wave_drag_fuselage(conditions,configuration,body,wing):
    """ SUAVE.Methods.wave_drag_fuselage(conditions,configuration,body,wing)
        computes the volume wave drag 
        
        Inputs:
        - body - body of revolution
        - total_length - length of body
        - Rmax - largest radius of body
        - Sref - main wing reference area

        Outputs:
        - CD due to volume wave drag

        Assumptions:
        - Current function designed for fuselage data
        
    """

    # Unpack
    total_length = body.length_total
    Rmax = body.Deff/2.0
    Sref = wing.sref

    
    # Computations - takes drag of Sears-Haack and uses wing reference area for CD
    wave_drag_body_of_rev = (9.0*(np.pi)**3.0*Rmax**4.0/(4.0*total_length**2.0))/(0.5*Sref)

    
    return wave_drag_body_of_rev*1.15


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