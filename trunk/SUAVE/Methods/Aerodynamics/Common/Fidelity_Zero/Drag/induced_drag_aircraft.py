## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Drag
# induced_drag_aircraft.py
# 
# Created:  Dec 2013, SUAVE Team
# Modified: Jan 2016, E. Botero
       

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
from SUAVE.Core import Data

# package imports
import numpy as np

# ----------------------------------------------------------------------
#  Induced Drag Aircraft
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Drag
def induced_drag_aircraft(state,settings,geometry):
    """Determines induced drag for the full aircraft

    Assumptions:
    Based on fits

    Source:
    adg.stanford.edu (Stanford AA241 A/B Course Notes)

    Inputs:
    state.conditions.aerodynamics.lift_coefficient               [Unitless]
    state.conditions.aerodynamics.drag_breakdown.parasite.total  [Unitless]
    configuration.oswald_efficiency_factor                       [Unitless]
    configuration.viscous_lift_dependent_drag_factor             [Unitless]
    geometry.wings['main_wing'].span_efficiency                  [Unitless]
    geometry.wings['main_wing'].aspect_ratio                     [Unitless]

    Outputs:
    total_induced_drag                                           [Unitless]

    Properties Used:
    N/A
    """

    # unpack inputs
    conditions    = state.conditions
    configuration = settings 
    
    CL            = conditions.aerodynamics.lift_coefficient
    K             = configuration.viscous_lift_dependent_drag_factor
    CDp           = state.conditions.aerodynamics.drag_breakdown.parasite.total
    CDi           = conditions.aerodynamics.drag_breakdown.induced.total    
    
    # compute span efficiency factor from invisid calculations
    e = (CDi*np.pi*ar)/(CL**2)
    
    # compute osward efficiency factor
    if e0 == None:
        e0 = 1/((1/e)+np.pi*ar*K*CDp)
   
    configuration.oswald_efficiency_factor = e0 
    conditions.aerodynamics.drag_breakdown.induced.span_efficiency_factor = e

    return CDi