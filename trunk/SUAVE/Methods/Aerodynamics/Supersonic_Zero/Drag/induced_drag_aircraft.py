## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
# induced_drag_aircraft.py
# 
# Created:  Feb 2019, T. MacDonald
# Modified: Jan 2020, T. MacDonald
     
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data

import numpy as np
from SUAVE.Methods.Utilities.Cubic_Spline_Blender import Cubic_Spline_Blender

# ----------------------------------------------------------------------
#  Induced Drag Aicraft
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
def induced_drag_aircraft(state,settings,geometry):
    """Determines induced drag for the full aircraft

    Assumptions:
    Based on fits

    Source:
    http://aerodesign.stanford.edu/aircraftdesign/aircraftdesign.html (Stanford AA241 A/B Course Notes)

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
    conditions = state.conditions
    configuration = settings    
    
    aircraft_lift = conditions.aerodynamics.lift_coefficient
    mach          = conditions.freestream.mach_number
    
    e             = configuration.oswald_efficiency_factor
    wing_e        = configuration.span_efficiency
    K             = configuration.viscous_lift_dependent_drag_factor
    ar            = geometry.wings['main_wing'].aspect_ratio 
    CDp           = state.conditions.aerodynamics.drag_breakdown.parasite.total
    
    if e == None:
        e = 1/((1/wing_e)+np.pi*ar*K*CDp)    
        
    spline = Cubic_Spline_Blender(.91,.99)
    h00 = lambda M:spline.compute(M)      
    
    total_induced_drag_low  = aircraft_lift**2 / (np.pi*ar*e)
    total_induced_drag_high = aircraft_lift**2 / (np.pi*ar*wing_e) # oswald factor would include wave drag due to lift
                                                                                    # which is not computed here
                                                                                    
    total_induced_drag      = total_induced_drag_low*h00(mach) + total_induced_drag_high*(1-h00(mach))
        
    # store data
    try:
        conditions.aerodynamics.drag_breakdown.induced = Data(
            total             = total_induced_drag ,
            efficiency_factor = e                  ,
            aspect_ratio      = ar                 ,
        )
    except:
        print("Drag Polar Mode")     
    
    return total_induced_drag