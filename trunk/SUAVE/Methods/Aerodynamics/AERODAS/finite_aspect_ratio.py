## @ingroup Methods-Aerodynamics-AERODAS
# finite_aspect_ratio.py
# 
# Created:  Feb 2016, E. Botero
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Core import Units

# ----------------------------------------------------------------------
#  Finite Aspect Ratio
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-AERODAS
def finite_aspect_ratio(state,settings,geometry):
    """Uses the AERODAS method to prestall parameters for lift and drag.

    Assumptions:
    None

    Source:
    NASA TR: "Models of Lift and Drag Coefficients of Stalled and Unstalled Airfoils in
      Wind Turbines and Wind Tunnels" by D. A. Spera

    Inputs:
    geometry.
      aspect_ratio                                [Unitless]
      section.
        maximum_coefficient_lift                  [Unitless]
        angle_attack_max_prestall_lift            [radians]
        pre_stall_maximum_drag_coefficient_angle  [radians]
        pre_stall_maximum_drag_coefficient        [Unitless]
    settings.section_lift_curve_slope             [radians]

    Outputs:
    pre_stall_maximum_lift_coefficient            [Unitless]
    pre_stall_maximum_lift_drag_coefficient       [Unitless]
    pre_stall_lift_curve_slope                    [radians]
    pre_stall_maximum_drag_coefficient_angle      [Unitless]
    (these are also packed into geometry.)

    Properties Used:
    N/A
    """      
    
    # unpack inputs
    wing    = geometry
    AR      = wing.aspect_ratio
    CL1maxp = wing.section.maximum_coefficient_lift  
    ACL1p   = wing.section.angle_attack_max_prestall_lift 
    ACD1p   = wing.section.pre_stall_maximum_drag_coefficient_angle
    CD1maxp = wing.section.pre_stall_maximum_drag_coefficient  
    S1p     = settings.section_lift_curve_slope / Units.deg
    
    # Equation 5a
    ACL1   = ACL1p + 18.2*CL1maxp*(AR**(-0.9)) * Units.deg
    
    # Equation 5b
    #S1     = S1p/(1+18.2*S1p*(AR**(-0.9))) * Units.deg
    
    # From McCormick
    S1 = S1p*AR/(2+np.sqrt(4+AR**2)) * Units.deg
    
    # Equation 5c
    ACD1   =  ACD1p + 18.2*CL1maxp*(AR**(-0.9)) * Units.deg
    
    # Equation 5d
    CD1max = CD1maxp + 0.280*(CL1maxp*CL1maxp)*(AR**(-0.9))
    
    # Equation 5e
    CL1max = CL1maxp*(0.67+0.33*np.exp(-(4.0/AR)**2.))

    # Pack outputs
    wing.pre_stall_maximum_lift_coefficient       = CL1max
    wing.pre_stall_maximum_lift_drag_coefficient  = CD1max
    wing.pre_stall_lift_curve_slope               = S1
    wing.pre_stall_maximum_drag_coefficient_angle = ACD1
    
    return CL1max, CD1max, S1, ACD1