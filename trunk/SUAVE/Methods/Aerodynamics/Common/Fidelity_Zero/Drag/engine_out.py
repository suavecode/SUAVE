## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Drag
# engine_out.py
#
# Created:  Apr 2020, T. MacDonald
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Helper_Functions.asymmetry_drag import \
     asymmetry_drag
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Helper_Functions import windmilling_drag

# ----------------------------------------------------------------------
#  Adds drag due to engine out
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Drag
def engine_out(state, settings, geometry):
    """Adds an engine out drag increment
    
    Assumptions:
    None
    
    Source:
    None
    
    Inputs:
    state                  <Data>
    settings.engine_out    <bool>
    geometry               <Data>
    
    Outputs:
    engine_out_drag        [Unitless]
    
    Properties Used:
    N/A
    """    

    # unpack inputs
    conditions     = state.conditions

    # various drag components
    engine_out_flag = settings.engine_out

    engine_out_drag = 0

    if engine_out_flag:
        windmilling_drag_coefficient = windmilling_drag(geometry,state)
        asym_drag_coef = asymmetry_drag(state, geometry, 
                                        windmilling_drag_coefficient = windmilling_drag_coefficient)
        engine_out_drag = engine_out_drag + windmilling_drag_coefficient + asym_drag_coef

    # untrimmed drag
    conditions.aerodynamics.drag_breakdown.engine_out_drag = engine_out_drag

    return engine_out_drag