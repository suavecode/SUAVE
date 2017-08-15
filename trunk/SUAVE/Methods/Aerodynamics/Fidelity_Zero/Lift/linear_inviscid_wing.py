## @ingroup Methods-Aerodynamics-Fidelity_Zero-Lift
# linear_inviscid_wing.py
# 
# Created:  Dec 2013, A. Variyar 
# Modified: Feb 2014, A. Variyar, T. Lukaczyk, T. Orra 
#           Apr 2014, A. Variyar
#           Jan 2015, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Linear Inviscid Wing
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Fidelity_Zero-Lift
def linear_inviscid_wing(state,settings,geometry):
    """Computes wing lift base on simple linear theory

    Assumptions:
    Linear airfoil theory

    Source:
    Linear airfoil theory

    Inputs:
    state.conditions.freestream.mach_number        [Unitless]
    state.conditions.aerodynamics.angle_of_attack  [Unitless]

    Outputs:
    wings_lift                                     [Unitless]

    Properties Used:
    N/A
    """             

    # unpack
    Mc             = state.conditions.freestream.mach_number
    AoA            = state.conditions.aerodynamics.angle_of_attack
    
    # inviscid lift of wings only
    inviscid_wings_lift = 2*np.pi*AoA 
    state.conditions.aerodynamics.lift_breakdown.inviscid_wings_lift.total = inviscid_wings_lift
         
    wings_lift = state.conditions.aerodynamics.lift_breakdown.inviscid_wings_lift.total
    
    state.conditions.aerodynamics.lift_coefficient= wings_lift

    return wings_lift