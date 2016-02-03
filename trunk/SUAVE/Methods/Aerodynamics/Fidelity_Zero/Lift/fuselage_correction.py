# fuselage_correction.py
# 
# Created:  Dec 2013, A. Variyar 
# Modified: Feb 2014, A. Variyar, T. Lukaczyk, T. Orra 
#           Apr 2014, A. Variyar
#           Jan 2015, E. Botero

# ----------------------------------------------------------------------
#  Fuselage Correction
# ----------------------------------------------------------------------

def fuselage_correction(state,settings,geometry):  
   
    # unpack
    fus_correction  = settings.fuselage_lift_correction
    Mc              = state.conditions.freestream.mach_number
    AoA             = state.conditions.aerodynamics.angle_of_attack
    wings_lift_comp = state.conditions.aerodynamics.lift_coefficient
    
    # total lift, accounting one fuselage
    aircraft_lift_total = wings_lift_comp * fus_correction 

    state.conditions.aerodynamics.lift_coefficient= aircraft_lift_total

    return aircraft_lift_total