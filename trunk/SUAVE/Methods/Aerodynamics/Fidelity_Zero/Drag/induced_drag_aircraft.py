
# induced_drag_aircraft.py
# 
# Created:  Your Name, Dec 2013
# Modified:         

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
from SUAVE.Attributes.Results import Result

# python imports
import os, sys, shutil
from copy import deepcopy
from warnings import warn

# package imports
import numpy as np
import scipy as sp


# ----------------------------------------------------------------------
#  The Function
# ----------------------------------------------------------------------

def induced_drag_aircraft(conditions,configuration,geometry):
    """ SUAVE.Methods.induced_drag_aircraft(conditions,configuration,geometry)
        computes the induced drag associated with a wing 
        
        Inputs:
        
        Outputs:
        
        Assumptions:
            based on a set of fits
            
    """

    # unpack inputs
    aircraft_lift = conditions.aerodynamics.lift_coefficient
    e             = configuration.aircraft_span_efficiency_factor # TODO: get estimate from weissinger
    ar            = geometry.wings[0].aspect_ratio # TODO: get estimate from weissinger
    
    # start the result
    total_induced_drag = 0.0
    
    #print("In induced_drag_aircraft:")
    #print aircraft_lift
    total_induced_drag = aircraft_lift**2 / (np.pi*ar*e)
    #raw_input()
        
    # store data
    conditions.aerodynamics.drag_breakdown.induced = Result(
        total             = total_induced_drag ,
        efficiency_factor = e                  ,
        aspect_ratio      = ar                 ,
    )
    
    # done!

    return total_induced_drag