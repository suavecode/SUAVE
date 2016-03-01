# compute_aircraft_drag.py
# 
# Created:  Dec 2013, A. Variyar
# Modified: Aug 2014, T. MacDonald
#           Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Methods.Aerodynamics.Supersonic_Zero.Drag import \
     parasite_drag_aircraft, compressibility_drag_total, \
     miscellaneous_drag_aircraft

# ----------------------------------------------------------------------
#  The Function
# ----------------------------------------------------------------------

def compute_aircraft_drag(conditions,configuration,geometry=None):
    """ SUAVE.Methods.Aerodynamics.compute_aircraft_drag_supersonic(conditions,configuration,geometry)
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
    trim_correction_factor = configuration.trim_drag_correction_factor
    drag_breakdown = conditions.aerodynamics.drag_breakdown
    
    # Various drag components
    parasite_total        = parasite_drag_aircraft(conditions,configuration,geometry) 
    induced_total         = induced_drag_aircraft(conditions,configuration,geometry)
    compressibility_total = compressibility_drag_total(conditions,configuration,geometry) 
    miscellaneous_drag    = miscellaneous_drag_aircraft(conditions,configuration,geometry)
    
    
    # Untrimmed drag
    aircraft_untrimmed = parasite_total        \
                       + induced_total         \
                       + compressibility_total \
                       + miscellaneous_drag 
    
    # Trim correction
    aircraft_total_drag = aircraft_untrimmed * trim_correction_factor
    drag_breakdown.miscellaneous.trim_correction_factor = trim_correction_factor
    
    # Store to results
    drag_breakdown.total     = aircraft_total_drag
    drag_breakdown.untrimmed = aircraft_untrimmed       
    
    return aircraft_total_drag