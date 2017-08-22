## @ingroup Methods-Aerodynamics-SU2_Euler
# total_aircraft_drag.py
# 
# Created:  Dec 2013, A. Variyar
# Modified: Oct 2016, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Drag import \
     induced_drag_aircraft, compressibility_drag_wing, \
     miscellaneous_drag_aircraft

# ----------------------------------------------------------------------
#  Total Aircraft
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-SU2_Euler
def total_aircraft_drag(state,settings,geometry):
    """ This computes the total drag of an aircraft and stores
    that data in the conditions structure.

    Assumptions:
    None

    Source:
    N/A

    Inputs:
    settings.drag_coefficient_increment            [Unitless]
    state.conditions.aerodynamics.drag_breakdown.
      trim_corrected_drag                          [Unitless]
      spoiler_drag                                 [Unitless]
      
    Outputs:
    aircraft_total_drag                            [Unitless]

    Properties Used:
    N/A
    """    
    
    # Unpack inputs
    conditions    = state.conditions
    configuration = settings
    
    drag_coefficient_increment = configuration.drag_coefficient_increment
    trim_corrected_drag        = conditions.aerodynamics.drag_breakdown.trim_corrected_drag
    spoiler_drag               = conditions.aerodynamics.drag_breakdown.spoiler_drag 

    aircraft_total_drag = 0.0
    # Add drag_coefficient_increment
    aircraft_total_drag += trim_corrected_drag + drag_coefficient_increment + spoiler_drag
    conditions.aerodynamics.drag_breakdown.drag_coefficient_increment = drag_coefficient_increment

    # Store to results
    conditions.aerodynamics.drag_breakdown.total = aircraft_total_drag
    conditions.aerodynamics.drag_coefficient     = aircraft_total_drag
    
    return aircraft_total_drag