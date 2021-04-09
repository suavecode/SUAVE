## @ingroup Methods-Aerodynamics-AERODAS
# post_stall_coefficients.py
# 
# Created:  Feb 2016, E. Botero
# Modified: Jun 2017, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Core import Units, Data

# ----------------------------------------------------------------------
#  Post Stall Coefficients
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-AERODAS
def post_stall_coefficients(state,settings,geometry):
    """Uses the AERODAS method to determine poststall parameters for lift and drag for a single wing

    Assumptions:
    None

    Source:
    NASA TR: "Models of Lift and Drag Coefficients of Stalled and Unstalled Airfoils in
      Wind Turbines and Wind Tunnels" by D. A. Spera

    Inputs:
    settings.section_zero_lift_angle_of_attack      [radians]
    geometry.
      aspect_ratio                                  [Unitless]
      thickness_to_chord                            [Unitless]
      section.angle_attack_max_prestall_lift        [radians]
      pre_stall_maximum_lift_drag_coefficient       [Unitless]
      pre_stall_maximum_drag_coefficient_angle      [Unitless]
    state.conditions.aerodynamics.angle_of_attack   [radians]
      

    Outputs:
    CL2 (coefficient of lift)                       [Unitless]
    CD2 (coefficient of drag)                       [Unitless]
    (packed in state.conditions.aerodynamics.post_stall_coefficients[geometry.tag])

    Properties Used:
    N/A
    """  
    
    # unpack inputs
    wing   = geometry
    A0     = settings.section_zero_lift_angle_of_attack
    AR     = wing.aspect_ratio
    t_c    = wing.thickness_to_chord
    ACL1   = wing.section.angle_attack_max_prestall_lift 
    CD1max = wing.pre_stall_maximum_lift_drag_coefficient
    ACD1   = wing.pre_stall_maximum_drag_coefficient_angle
    alpha  = state.conditions.aerodynamics.angle_of_attack
    
    if wing.vertical == True:
        alpha = 0. * np.ones_like(alpha)    
            
    # Equation 9a and b
    F1        = 1.190*(1.0-(t_c*t_c))
    F2        = 0.65 + 0.35*np.exp(-(9.0/AR)**2.3)
    
    # Equation 10b and c
    G1        = 2.3*np.exp(-(0.65*t_c)**0.9)
    G2        = 0.52 + 0.48*np.exp(-(6.5/AR)**1.1)
    
    # Equation 8a and b
    CL2max    = F1*F2
    CD2max    = G1*G2
    
    # Equation 11d
    RCL2      = 1.632-CL2max
    
    # Equation 11e
    N2        = 1 + CL2max/RCL2
    
    # Equation 11a,b,c
    con1      = np.logical_and(0<alpha,alpha<ACL1)
    con2      = np.logical_and(ACL1<=alpha,alpha<=(92.0*Units.deg))
    con3      = [alpha>=(92.0*Units.deg)]
    CL2       = np.zeros_like(alpha)
    CL2[con1] =  0
    CL2[con2] = -0.032*(alpha[con2]/Units.deg-92.0) - RCL2*((92.*Units.deg-alpha[con2])/(51.0*Units.deg))**N2
    CL2[con3] = -0.032*(alpha[con3]/Units.deg-92.0) + RCL2*((alpha[con3]-92.*Units.deg)/(51.0*Units.deg))**N2
    
    # If alpha is negative flip things for lift
    alphan    = - alpha+2*A0
    con1      = np.logical_and(0<alphan, alphan<ACL1)
    con2      = np.logical_and(ACL1<=alphan, alphan<=(92.0*Units.deg))
    con3      = alphan>=(92.0*Units.deg)
    CL2[con1] = 0.
    CL2[con2] = 0.032*(alphan[con2]/Units.deg-92.0) + RCL2*((92.*Units.deg-alphan[con2])/(51.0*Units.deg))**N2
    CL2[con3] = 0.032*(alphan[con3]/Units.deg-92.0) - RCL2*((alphan[con3]-92.*Units.deg)/(51.0*Units.deg))**N2
    
    # Equation 12a 
    con1      = np.logical_and((2*A0-ACL1)<alpha, alpha<ACL1)
    con2      = alpha>ACD1
    CD2       = 0.0 * np.ones_like(alpha)
    CD2[con1] = 0.
    CD2[con2] = CD1max[con2] + (CD2max - CD1max[con2]) * np.sin((alpha[con2]-ACD1[con2])/(np.pi/2-ACD1[con2]))
    
    # If alpha is negative flip things for drag
    alphan    = -alpha + 2*A0
    con1      = np.logical_and((2*A0-ACL1)<alphan,alphan<ACL1)
    con2      = alphan>=ACD1
    CD2[con1] = 0.
    CD2[con2] = CD1max[con2] + (CD2max - CD1max[con2]) * np.sin((alphan[con2]-ACD1[con2])/(np.pi/2-ACD1[con2]))        
        
    # Pack outputs
    wing_result = Data(
        lift_coefficient = CL2,
        drag_coefficient = CD2
        )
    
    state.conditions.aerodynamics.post_stall_coefficients[wing.tag] =  wing_result
    
    return CL2, CD2