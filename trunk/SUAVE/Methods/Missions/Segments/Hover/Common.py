## @ingroup Methods-Missions-Segments-Hover
# Common.py
# 
# Created:  Jan 2016, E. Botero
# Modified:

# ----------------------------------------------------------------------
#  Unpack Unknowns
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Hover
def unpack_unknowns(segment,state):
    """ Unpacks the throttle setting from the solver to the mission
    
        Assumptions:
        N/A
        
        Inputs:
            state.unknowns:
                throttle    [Unitless]
            
        Outputs:
            state.conditions:
                propulsion.throttle            [Unitless]

        Properties Used:
        N/A
                                
    """     
    
    # unpack unknowns
    throttle   = state.unknowns.throttle
    
    # apply unknowns
    state.conditions.propulsion.throttle[:,0] = throttle[:,0]
    

# ----------------------------------------------------------------------
#  Residual Total Forces
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Hover
def residual_total_forces(segment,state):
    """ Calculates a residual based on forces
    
        Assumptions:
        The vehicle is not accelerating, doesn't use gravity. Only vertical forces
        
        Inputs:
            state.conditions:
                frames.inertial.total_force_vector [Newtons]
            
        Outputs:
            state.residuals.forces [meters/second^2]

        Properties Used:
        N/A
                                
    """            
    
    FT = state.conditions.frames.inertial.total_force_vector

    # vertical
    state.residuals.forces[:,0] = FT[:,2]

    return
    
    
 
    
    