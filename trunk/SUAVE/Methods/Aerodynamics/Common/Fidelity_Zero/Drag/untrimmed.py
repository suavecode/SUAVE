## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Drag
# untrimmed.py
#
# Created:  Jan 2014, T. Orra
# Modified: Jan 2016, E. Botero  

# ----------------------------------------------------------------------
#  Computes the miscellaneous drag
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Drag
def untrimmed(state,settings,geometry):
    """Sums aircraft drag before trim correction

    Assumptions:
    None

    Source:
    None

    Inputs:
    state.conditions.aerodynamics.drag_breakdown.
      parasite.total                              [Unitless]
      induced.total                               [Unitless]
      compressible.total                          [Unitless]
      miscellaneous.total                         [Unitless]

    Outputs:
    aircraft_untrimmed                            [Unitless]

    Properties Used:
    N/A
    """       

    # unpack inputs
    conditions     = state.conditions

    # various drag components
    parasite_total        = conditions.aerodynamics.drag_breakdown.parasite.total            
    induced_total         = conditions.aerodynamics.drag_breakdown.induced.total            
    compressibility_total = conditions.aerodynamics.drag_breakdown.compressible.total         
    miscellaneous_drag    = conditions.aerodynamics.drag_breakdown.miscellaneous.total 

    # untrimmed drag
    aircraft_untrimmed = parasite_total        \
                       + induced_total         \
                       + compressibility_total \
                       + miscellaneous_drag
    
    conditions.aerodynamics.drag_breakdown.untrimmed = aircraft_untrimmed
    
    return aircraft_untrimmed
