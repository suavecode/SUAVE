## @ingroup Methods-Aerodynamics-Fidelity_Zero-Drag
# total_aircraft_drag.py
# 
# Created:  Dec 2013, A. Variyar
# Modified: Feb 2014, A. Variyar, T. Lukaczyk, T. Orra
#           Jan 2016, E. Botero 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
from SUAVE.Analyses import Results

from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Drag import \
     induced_drag_aircraft, compressibility_drag_wing, \
     miscellaneous_drag_aircraft

# ----------------------------------------------------------------------
#  Total Aircraft
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Fidelity_Zero-Drag
def total_aircraft(state,settings,geometry):
    """Computes the total drag for an aircraft

    Assumptions:
    None

    Source:
    None

    Inputs:
    state.conditions.aerodynamics.drag_breakdown.
      trim_corrected_drag                    [Unitless]
      spoiler_drag                           [Unitless]
    settings.drag_coefficient_increment      [Unitless]

    Outputs:
    aircraft_total_drag                      [Unitless]

    Properties Used:
    N/A
    """    
    
    # unpack inputs
    conditions    = state.conditions
    configuration = settings
    
    drag_coefficient_increment = configuration.drag_coefficient_increment
    trim_corrected_drag        = conditions.aerodynamics.drag_breakdown.trim_corrected_drag
    spoiler_drag               = conditions.aerodynamics.drag_breakdown.spoiler_drag

    aircraft_total_drag = 0.0
    # add drag_coefficient_increment
    aircraft_total_drag += trim_corrected_drag + drag_coefficient_increment + spoiler_drag
    conditions.aerodynamics.drag_breakdown.drag_coefficient_increment = drag_coefficient_increment

    # store to results
    conditions.aerodynamics.drag_breakdown.total = aircraft_total_drag
    conditions.aerodynamics.drag_coefficient     = aircraft_total_drag
    
    return aircraft_total_drag