## @ingroup Methods-Aerodynamics-Supersonic_Zero-Lift
# wing_compressibility.py
# 
# Created:  Dec 2013, A. Variyar 
# Modified: Feb 2014, A. Variyar, T. Lukaczyk, T. Orra 
#           Apr 2014, A. Variyar
#           Aug 2014, T. Macdonald
#           Jan 2015, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Wing Compressibility
# ----------------------------------------------------------------------
## @ingroup Methods-Aerodynamics-Supersonic_Zero-Lift
def wing_compressibility(state,settings,geometry):
    """Corrects a wings lift based on compressibility, allow supersonic mach number

    Assumptions:
    wing capable of vortex lift

    Source:
    https://stanford.edu/~cantwell/AA200_Course_Material/AA200_Course_Notes/

    Inputs:
    settings.fuselage_lift_correction  [-]
    state.conditions.
      freestream.mach_number           [-]
      aerodynamics.angle_of_attack     [radians]
      aerodynamics.lift_coefficient    [-]

    Outputs:
    state.conditions.aerodynamics.
      lift_breakdown.compressible_wings [-] CL for the wings
      lift_coefficient                  [-]
    wings_lift_comp                     [-]

    Properties Used:
    N/A
    """  
    
    # unpack
    fus_correction = settings.fuselage_lift_correction
    Mc             = state.conditions.freestream.mach_number
    AoA            = state.conditions.aerodynamics.angle_of_attack
    
    wings_lift = state.conditions.aerodynamics.lift_coefficient
    
    # compressibility correction
    compress_corr = np.array([[0.0]] * len(Mc))
    compress_corr[Mc < 0.95] = 1./(np.sqrt(1.-Mc[Mc < 0.95]**2.))
    compress_corr[Mc >= 0.95] = 1./(np.sqrt(1.-0.95**2)) # Values for Mc > 1.05 are update after this assignment 
    compress_corr[Mc > 1.05] = 1./(np.sqrt(Mc[Mc > 1.05]**2.-1.))

    # correct lift
    wings_lift_comp = wings_lift * compress_corr  
    
    state.conditions.aerodynamics.lift_breakdown.compressible_wings = wings_lift_comp
    state.conditions.aerodynamics.lift_coefficient= wings_lift_comp    
    
    return wings_lift_comp