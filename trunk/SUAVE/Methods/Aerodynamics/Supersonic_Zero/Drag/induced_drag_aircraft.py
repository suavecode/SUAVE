# induced_drag_aircraft.py
# 
# Created:  Aug 2014, T. MacDonald
# Modified: Nov 2016, T. MacDonald
     
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Analyses import Results

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
    Mc            = conditions.freestream.mach_number[:,0]
    
    # unclear how valid this is for the supersonic condition, needs to be checked
    e             = configuration.oswald_efficiency_factor
    K             = configuration.viscous_lift_dependent_drag_factor
    wing_e        = geometry.wings['main_wing'].span_efficiency
    ar            = geometry.wings['main_wing'].aspect_ratio 
    CDp           = state.conditions.aerodynamics.drag_breakdown.parasite.total
    
    if e == None:
        e = 1/((1/wing_e)+np.pi*ar*K*CDp)    
    
    # start the results
    total_induced_drag = np.array([[0.0]]*len(Mc))
    total_induced_drag[Mc < 1.0] = aircraft_lift[Mc < 1.0]**2 / (np.pi*ar*e[Mc < 1.0])
    total_induced_drag[Mc >= 1.0] = aircraft_lift[Mc >= 1.0]**2 / (np.pi*ar*e[Mc >= 1.0]) # for future changes to e
        
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