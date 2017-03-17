## @ingroup SU2_Euler
# untrimmed.py
#
# Created:  Jan 2014, T. Orra
# Modified: Oct 2016, T. MacDonald 

import numpy as np # should be removed, need to determine how to handle this so create by dates dont appear

## @ingroup SU2_Euler
def untrimmed(state,settings,geometry):
    """ This computes the total drag of an aircraft without trim
    and stores that data in the conditions structure.

    Assumptions:
    None

    Source:
    N/A

    Inputs:
    state.conditions.aerodynamics.drag_breakdown.
      parasite.total                               [Unitless]
      induced.total                                [Unitless]
      compressible.total                           [Unitless]
      miscellaneous.total                          [Unitless]

    Outputs:
    aircraft_untrimmed                             [Unitless]

    Properties Used:
    N/A
    """      

    # Unpack inputs
    conditions     = state.conditions
    configuration  = settings
    drag_breakdown = conditions.aerodynamics.drag_breakdown

    # Various drag components
    parasite_total        = conditions.aerodynamics.drag_breakdown.parasite.total            
    induced_total         = conditions.aerodynamics.drag_breakdown.induced.total            
    compressibility_total = conditions.aerodynamics.drag_breakdown.compressible.total         
    miscellaneous_drag    = conditions.aerodynamics.drag_breakdown.miscellaneous.total 

    # Untrimmed drag
    aircraft_untrimmed = parasite_total        \
        + induced_total         \
        + compressibility_total \
        + miscellaneous_drag
    
    conditions.aerodynamics.drag_breakdown.untrimmed = aircraft_untrimmed
    
    return aircraft_untrimmed
