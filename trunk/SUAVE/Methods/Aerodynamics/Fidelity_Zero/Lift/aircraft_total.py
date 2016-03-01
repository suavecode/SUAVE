# aircraft_total.py
# 
# Created:  Dec 2013, A. Variyar,
# Modified: Feb 2014, A. Variyar, T. Lukaczyk, T. Orra 
#           Apr 2014, A. Variyar   
#           Jan 2016, E. Botero       


# ----------------------------------------------------------------------
#  Aircraft Total
# ----------------------------------------------------------------------

def aircraft_total(state,settings,geometry):

    # unpack
    fus_correction = settings.fuselage_lift_correction
    Mc             = state.conditions.freestream.mach_number
    AoA            = state.conditions.aerodynamics.angle_of_attack
    
    aircraft_lift_total = state.conditions.aerodynamics.lift_coefficient
    state.conditions.aerodynamics.lift_coefficient= aircraft_lift_total

    return aircraft_lift_total
