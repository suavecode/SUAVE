# AERODAS_setup.py
# 
# Created:  Feb 2016, E. Botero
# Modified: Jun 2017, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Core import Data, Units

# ----------------------------------------------------------------------
#  Setup Daa
# ----------------------------------------------------------------------

def setup_data(state,settings,geometry):
    """This model is based on the NASA TR: "Models of Lift and Drag Coefficients of Stalled and Unstalled Airfoils in
     Wind Turbines and Wind Tunnels" by D. A. Spera
     
     Setup"""
    
    state.conditions.aerodynamics.pre_stall_coefficients  = Data()
    state.conditions.aerodynamics.post_stall_coefficients = Data()
    
    
    return 

# ----------------------------------------------------------------------
#  Lift and Drag Total
# ----------------------------------------------------------------------

def lift_drag_total(state,settings,geometry):
    """This model is based on the NASA TR: "Models of Lift and Drag Coefficients of Stalled and Unstalled Airfoils in
     Wind Turbines and Wind Tunnels" by D. A. Spera
     
     Sum up all contributions from the wings"""
    
    # prime the totals
    CL_total = 0.
    CD_total = 0.
    
    # Unpack general things
    ref       = geometry.reference_area
    wing_aero = state.conditions.aerodynamics
    alpha     = state.conditions.aerodynamics.angle_of_attack
    A0        = settings.section_zero_lift_angle_of_attack
    
    #  loop through each wing 
    for wing in geometry.wings:
        
        # unpack inputs
        area = wing.areas.reference
        CL1  = wing_aero.pre_stall_coefficients[wing.tag].lift_coefficient
        CD1  = wing_aero.pre_stall_coefficients[wing.tag].drag_coefficient
        CL2  = wing_aero.post_stall_coefficients[wing.tag].lift_coefficient
        CD2  = wing_aero.post_stall_coefficients[wing.tag].drag_coefficient
        
        # Equation 3a
        CL = np.fmax(CL1,CL2)
        
        # Equation 3b
        CL[alpha<=A0] = np.fmin(CL1[alpha<=A0],CL2[alpha<=A0])
        
        # Equation 3c
        CD            = np.fmax(CD1,CD2)
        
        # Add to the total
        CD_total      = CD_total + CD*area/ref

        if wing.vertical == False:
            CL_total      = CL_total + CL*area/ref
        else:
            pass

        
    CD_total = CD_total + settings.drag_coefficient_increment
        
    # Pack outputs
    state.conditions.aerodynamics.lift_coefficient = CL_total
    state.conditions.aerodynamics.drag_coefficient = CD_total
    
    return CL_total, CD_total

# ----------------------------------------------------------------------
#  Lift Total
# ----------------------------------------------------------------------

def lift_total(state,settings,geometry):
    """"""
    
    CL = state.conditions.aerodynamics.lift_coefficient     

    return CL

# ----------------------------------------------------------------------
#  Drag Total
# ----------------------------------------------------------------------

def drag_total(state,settings,geometry):
    """"""
    
    CD = state.conditions.aerodynamics.drag_coefficient     
    
    return CD