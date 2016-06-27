# induced_drag_aircraft.py
# 
# Created:  Aug 2014, T. Macdonald
# Modified: Jan 2016, E. Botero
     
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Results

import numpy as np

# ----------------------------------------------------------------------
#  Induced Drag Aicraft
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
    ar            = geometry.wings['main_wing'].aspect_ratio # TODO: get estimate from weissinger
    Mc            = conditions.freestream.mach_number
    
    # start the results
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
    
    return total_induced_drag