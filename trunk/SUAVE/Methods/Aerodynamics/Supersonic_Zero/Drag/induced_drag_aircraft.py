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
    e             = configuration.aircraft_span_efficiency_factor # TODO: get estimate from weissinger
    ar            = geometry.wings[0].aspect_ratio # TODO: get estimate from weissinger
    Mc            = conditions.freestream.mach_number
    #e             = geometry.wings[0].span_efficiency
    
    # start the result
    total_induced_drag = 0.0
    
    #print("In induced_drag_aircraft:")
    #print aircraft_lift
    #for ii in range(len(Mc)):
        #if Mc[ii] < 1.0:
            #total_induced_drag = aircraft_lift**2 / (np.pi*ar*e)
        #else:
            #total_induced_drag = aircraft_lift**2 / (np.pi*ar)
            ##total_induced_drag = aircraft_lift * 0.0
            
    total_induced_drag = np.array([[0.0]]*len(Mc))
    total_induced_drag[Mc < 1.0] = aircraft_lift[Mc < 1.0]**2 / (np.pi*ar*e)
    total_induced_drag[Mc >= 1.0] = aircraft_lift[Mc >= 1.0]**2 / (np.pi*ar*e)
        
    # store data
    try:
        conditions.aerodynamics.drag_breakdown.induced = Results(
            total             = total_induced_drag ,
            efficiency_factor = e                  ,
            aspect_ratio      = ar                 ,
        )
    except:
        print("Drag Polar Mode")     
    
    # done!
    return total_induced_drag