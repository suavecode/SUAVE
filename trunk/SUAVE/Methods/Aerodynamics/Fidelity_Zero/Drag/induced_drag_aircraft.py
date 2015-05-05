
# induced_drag_aircraft.py
# 
# Created:  Your Name, Dec 2013
# Modified:         

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
from SUAVE.Core import Results

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

#def induced_drag_aircraft(conditions,configuration,geometry):
def induced_drag_aircraft(state,settings,geometry):
    """ SUAVE.Methods.induced_drag_aircraft(conditions,configuration,geometry)
        computes the induced drag associated with a wing 
        
        Inputs:
        
        Outputs:
        
        Assumptions:
            based on a set of fits
            
    """

    # unpack inputs
    conditions = state.conditions
    configuration = settings
    
    
    aircraft_lift = conditions.aerodynamics.lift_coefficient
    e             = configuration.oswald_efficiency_factor
    K             = configuration.viscous_lift_dependent_drag_factor
    wing_e        = geometry.wings[0].span_efficiency
    ar            = geometry.wings[0].aspect_ratio # TODO: get estimate from weissinger
    CDp           = state.conditions.aerodynamics.drag_breakdown.parasite.total
    
    if e == None:
        e = 1/((1/wing_e)+np.pi*ar*K*CDp)
    
    
    # start the result
    total_induced_drag = 0.0
    
    #print("In induced_drag_aircraft:")
    #print aircraft_lift
    total_induced_drag = aircraft_lift**2 / (np.pi*ar*e)
    #raw_input()
        
    # store data
    conditions.aerodynamics.drag_breakdown.induced = Results(
        total             = total_induced_drag ,
        efficiency_factor = e                  ,
        aspect_ratio      = ar                 ,
    )
    
    # done!

    return total_induced_drag