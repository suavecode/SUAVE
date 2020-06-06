## @ingroup Methods-Aerodynamics-Supersonic_Zero-Lift
# vortex_lift.py
# 
# Created:  Jun 2014, T. MacDonald
# Modified: Jul 2014, T. MacDonald
#           Jan 2016, E. Botero
#           Aug 2018, T. MacDonald
#           Apr 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.convert_sweep import convert_sweep
# ----------------------------------------------------------------------
#   The Function
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Supersonic_Zero-Lift
def vortex_lift(state,settings,geometry):
    """Computes vortex lift according to the Polhamus Suction Analogy

    Assumptions:
    simple delta wing

    Source:
    http://aerodesign.stanford.edu/aircraftdesign/highlift/sstclmax.html
    
    Inputs:
    states.conditions.
      freestream.mach_number              [-]
      aerodynamics.angle_of_attack        [radians]
      aerodynamics.lift_coefficient       [-]
    geometry.wings.*.aspect_ratio         [Unitless]
    geometry.wings.*.sweeps.leading_edge  [radians]

    Outputs:
    state.conditions.aerodynamics.
      lift_breakdown.vortex_lift          [-] CL due to vortex lift
    wings_lift                            [-] Total CL at this point

    Properties Used:
    N/A
    """      



    Mc         = state.conditions.freestream.mach_number
    AoA        = state.conditions.aerodynamics.angle_of_attack
    wings_lift = np.zeros_like(state.conditions.aerodynamics.lift_coefficient)
    vortex_cl  = np.zeros_like(wings_lift)

    for wing in geometry.wings: 
        wing_lift = state.conditions.aerodynamics.lift_breakdown.inviscid_wings[wing.tag]

        if wing.vortex_lift is True:
            # compute leading edge sweek if not given
            if wing.sweeps.leading_edge == None:         
                gamma     = convert_sweep(wing,old_ref_chord_fraction = 0.25 ,new_ref_chord_fraction = 0.0)
            else:
                gamma      = wing.sweeps.leading_edge
                
            AR = wing.aspect_ratio
            a = AoA[Mc < 1.0]
            
            # Calculate vortex lift
            vortex_cl[Mc < 1.0] += np.pi*AR/2*np.sin(a)*np.cos(a)*(np.cos(a)+np.sin(a)*np.cos(a)/np.cos(gamma) - np.sin(a)/(2*np.cos(gamma)))
           
            # Apply to wing lift
            wing_lift[Mc < 1.0] = vortex_cl[Mc < 1.0]
        
        wings_lift += wing_lift
    
    state.conditions.aerodynamics.lift_coefficient           = wings_lift
    state.conditions.aerodynamics.lift_breakdown.vortex_lift = vortex_cl   
    
    return vortex_cl