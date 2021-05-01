## @ingroup Methods-Aerodynamics-AERODAS
# pre_stall_coefficients.py
# 
# Created:  Feb 2016, E. Botero
# Modified: Jun 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Core import Data

# ----------------------------------------------------------------------
#  Pre Stall Coefficients
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-AERODAS
def pre_stall_coefficients(state,settings,geometry):
    """Uses the AERODAS method to determine prestall parameters for lift and drag for a single wing

    Assumptions:
    None

    Source:
    NASA TR: "Models of Lift and Drag Coefficients of Stalled and Unstalled Airfoils in
      Wind Turbines and Wind Tunnels" by D. A. Spera

    Inputs:
    state.conditions.aerodynamics.angle_of_attack
    settings.section_zero_lift_angle_of_attack
    geometry.
      section.
        angle_attack_max_prestall_lift
        zero_lift_drag_coefficient
      pre_stall_maximum_drag_coefficient_angle
      pre_stall_maximum_lift_coefficient
      pre_stall_lift_curve_slope 
      pre_stall_maximum_lift_drag_coefficient

    Outputs:
    CL1 (coefficient of lift)                       [Unitless]
    CD1 (coefficient of drag)                       [Unitless]
    (packed in state.conditions.aerodynamics.pre_stall_coefficients[geometry.tag])

    Properties Used:
    N/A
    """  
    
    # unpack inputs
    wing   = geometry
    alpha  = state.conditions.aerodynamics.angle_of_attack * 1.0
    A0     = settings.section_zero_lift_angle_of_attack
    ACL1   = wing.section.angle_attack_max_prestall_lift 
    ACD1   = wing.pre_stall_maximum_drag_coefficient_angle
    CL1max = wing.pre_stall_maximum_lift_coefficient
    S1     = wing.pre_stall_lift_curve_slope  
    CD1max = wing.pre_stall_maximum_lift_drag_coefficient
    CDmin  = wing.section.minimum_drag_coefficient 
    ACDmin = wing.section.minimum_drag_coefficient_angle_of_attack 
    
    if wing.vertical == True:
        alpha = 0. * np.ones_like(alpha)
        
        
    
        
        
    # Equation 6c
    RCL1          = S1*(ACL1-A0)-CL1max
    RCL1[RCL1<=0] = 1.e-16
    
    # Equation 6d
    N1            = 1 + CL1max/RCL1
    
    # Equation 6a or 6b depending on the alpha
    CL1            = 0.0 * np.ones_like(alpha)
    CL1[alpha>A0]  = S1*(alpha[alpha>A0]-A0)-RCL1[alpha>A0]*((alpha[alpha>A0]-A0)/(ACL1[alpha>A0]-A0))**N1[alpha>A0]
    CL1[alpha==A0] = 0.0
    CL1[alpha<A0]  = S1*(alpha[alpha<A0]-A0)+RCL1[alpha<A0]*((A0-alpha[alpha<A0])/(ACL1[alpha<A0]-A0))**N1[alpha<A0]
    
    # M what is m?
    M              = 2.0 # Does this need changing

    # Equation 7a
    con      = np.logical_and((2*A0-ACD1)<=alpha,alpha<=ACD1)
    CD1      = np.ones_like(alpha)
    CD1[con] = CDmin[con] + (CD1max[con]-CDmin[con])*((alpha[con] - ACDmin)/(ACD1[con]-ACDmin))**M    
    
    # Equation 7b
    CD1[np.logical_not(con)] = 0.
    
    # Pack outputs
    wing_result = Data(
        lift_coefficient = CL1,
        drag_coefficient = CD1
        )

    state.conditions.aerodynamics.pre_stall_coefficients[wing.tag] =  wing_result
    

    return CL1, CD1