## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# fuselage_correction.py
# 
# Created:  Dec 2013, A. Variyar 
# Modified: Feb 2014, A. Variyar, T. Lukaczyk, T. Orra 
#           Apr 2014, A. Variyar
#           Jan 2015, E. Botero

# ----------------------------------------------------------------------
#  Fuselage Correction
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def fuselage_correction(state,settings,geometry):  
    """Corrects aircraft lift based on fuselage effects

    Assumptions:
    None

    Source:
    adg.stanford.edu (Stanford AA241 A/B Course Notes)

    Inputs:
    settings.fuselage_lift_correction  [Unitless]
    state.conditions.
      freestream.mach_number           [Unitless]
      aerodynamics.angle_of_attack     [radians]
      aerodynamics.lift_coefficient    [Unitless]

    Outputs:
    aircraft_lift_total                [Unitless]

    Properties Used:
    N/A
    """         
   
    # unpack
    fus_correction  = settings.fuselage_lift_correction
    wings_lift_comp = state.conditions.aerodynamics.lift_coefficient
    
    # total lift, accounting one fuselage
    aircraft_lift_total = wings_lift_comp * fus_correction 

    state.conditions.aerodynamics.lift_coefficient= aircraft_lift_total

    return aircraft_lift_total