""" untrimmed.py: Provides the drag coefficient before trimming. """
## @ingroup SU2_Euler
#
# Created:  Jan 2014, T. Orra
# Modified: Oct 2016, T. MacDonald 

pass

## @ingroup SU2_Euler
def untrimmed(state,settings,geometry):
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
    conditions     = state.conditions
    configuration  = settings
    drag_breakdown = conditions.aerodynamics.drag_breakdown

    # Various drag components
    parasite_total        = conditions.aerodynamics.drag_breakdown.parasite.total            
    induced_total         = conditions.aerodynamics.drag_breakdown.induced.total            
    compressibility_total = conditions.aerodynamics.drag_breakdown.compressible.total         
    miscellaneous_drag    = conditions.aerodynamics.drag_breakdown.miscellaneous.total 

    # Untrimmed drag
    aircraft_untrimmed = parasite_total        \
        + induced_total         \
        + compressibility_total \
        + miscellaneous_drag
    
    conditions.aerodynamics.drag_breakdown.untrimmed = aircraft_untrimmed
    
    return aircraft_untrimmed
