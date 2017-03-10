""" total_aircraft_drag.py: Provide a total drag coefficient. """
## @ingroup SU2_Euler
# 
# Created:  Dec 2013, A. Variyar
# Modified: Oct 2016, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
from SUAVE.Analyses import Results

from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Drag import \
     induced_drag_aircraft, compressibility_drag_wing, \
     miscellaneous_drag_aircraft

# ----------------------------------------------------------------------
#  Total Aircraft
# ----------------------------------------------------------------------

## @ingroup SU2_Euler
def total_aircraft_drag(state,settings,geometry):
    """ SUAVE.Methods.Aerodynamics.compute_aircraft_drag(conditions,configuration,geometry)
        computes the lift associated with an aircraft 
        
        Inputs:
            conditions - data dictionary with fields:
                mach_number - float or 1D array of freestream mach numbers
                angle_of_attack - floar or 1D array of angle of attacks
                
            configuration - data dictionary with fields:
                surrogate_models.lift_coefficient - a callable function or class 
                    with inputs of angle of attack and outputs of lift coefficent
                fuselage_lift_correction - the correction to fuselage contribution to lift
                    
            geometry - the aircraft geoemtry with fields:
            
        
        Outputs:
            CD - float or 1D array of drag coefficients of the total aircraft
        
        Updates:
            conditions.drag_breakdown - stores results here
            
        Assumptions:
            
            
    """    
    
    # Unpack inputs
    conditions    = state.conditions
    configuration = settings
    
    drag_coefficient_increment = configuration.drag_coefficient_increment
    trim_corrected_drag        = conditions.aerodynamics.drag_breakdown.trim_corrected_drag
    spoiler_drag               = conditions.aerodynamics.drag_breakdown.spoiler_drag 

    aircraft_total_drag = 0.0
    # Add drag_coefficient_increment
    aircraft_total_drag += trim_corrected_drag + drag_coefficient_increment + spoiler_drag
    conditions.aerodynamics.drag_breakdown.drag_coefficient_increment = drag_coefficient_increment

    # Store to results
    conditions.aerodynamics.drag_breakdown.total = aircraft_total_drag
    conditions.aerodynamics.drag_coefficient     = aircraft_total_drag
    
    return aircraft_total_drag