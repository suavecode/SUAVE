## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Drag
# induced_drag_aircraft.py
# 
# Created:  Dec 2013, SUAVE Team
# Modified: Jan 2016, E. Botero
#                     S. Karpuk
       

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
    M. Nita, D. Scholz, 'Estimating the Oswald Factor from Basic Aircraft Geometrical Parameters'
        hamburg University of Applied Sciences, Aero - Aircraft Design and Systems Group

    Inputs:
    state.conditions.aerodynamics.lift_coefficient               [Unitless]
    state.conditions.aerodynamics.drag_breakdown.parasite.total  [Unitless]
    configuration.oswald_efficiency_factor                       [Unitless]
    configuration.viscous_lift_dependent_drag_factor             [Unitless]
    geometry.wings['main_wing'].span_efficiency                  [Unitless]
    geometry.wings['main_wing'].aspect_ratio                     [Unitless]
    geometry.wings['main_wing'].spans.projected                  [m]
    geometry.wings['main_wing'].taper                            [Unitless]
    geometry.fuselages['fuselage'].width                         [m]

    Outputs:
    total_induced_drag                                           [Unitless]

    Properties Used:
    N/A
    """

    # unpack inputs
    conditions    = state.conditions
    configuration = settings
    
    
    aircraft_lift = conditions.aerodynamics.lift_coefficient
    e             = configuration.oswald_efficiency_factor
    K             = configuration.viscous_lift_dependent_drag_factor
    span          = geometry.wings['main_wing'].spans.projected 
    wing_e        = geometry.wings['main_wing'].span_efficiency
    ar            = geometry.wings['main_wing'].aspect_ratio 
    CDp           = state.conditions.aerodynamics.drag_breakdown.parasite.total
    taper         = geometry.wings['main_wing'].taper 
    
    if 'fuselage' in geometry.fuselages:
        d_f = geometry.fuselages['fuselage'].width
    else:
        d_f = 0
        
    if e == None:
        s     = 1 - 2 * (d_f/span)**2
        f     = 0.0524*taper**4 - 0.15*taper**3 + 0.1659*taper**2 - 0.0706*taper + 0.0119
        u     = 1/(1+f*ar)
        e     = 1/(1/(u*s)+np.pi*ar*K*CDp)
    
    # start the result
    total_induced_drag = aircraft_lift**2 / (np.pi*ar*e)
        
    # store data
    conditions.aerodynamics.drag_breakdown.induced = Data(
        total             = total_induced_drag ,
        efficiency_factor = e                  ,
        aspect_ratio      = ar                 ,
    )

    return total_induced_drag
