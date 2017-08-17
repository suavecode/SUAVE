## @ingroup Methods-Aerodynamics-Supersonic_Zero-Lift
# compute_aircraft_lift.py
# 
# Created:  Dec 2013, A. Variyar,
# Modified: Feb 2014, A. Variyar, T. Lukaczyk, T. Orra 
#           Apr 2014, A. Variyar  
#           Aug 2014, T. Macdonald
#           Jan 2016, E. Botero       

# ----------------------------------------------------------------------
#  Aircraft Total
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Supersonic_Zero-Lift
def aircraft_total(state,settings,geometry):
    """Returns total aircraft lift and stores values

    Assumptions:
    None

    Source:
    None

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
    fus_correction = settings.fuselage_lift_correction
    Mc             = state.conditions.freestream.mach_number
    AoA            = state.conditions.aerodynamics.angle_of_attack
    
    aircraft_lift_total = state.conditions.aerodynamics.lift_coefficient
    state.conditions.aerodynamics.lift_coefficient = aircraft_lift_total    

    return aircraft_lift_total