## @ingroup Methods-Missions-Segments-Cruise
# Common.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Unpack Unknowns
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Cruise
def unpack_unknowns(segment,state):
    """ Unpacks the throttle setting and body angle from the solver to the mission
    
        Assumptions:
        N/A
        
        Inputs:
            state.unknowns:
                throttle    [Unitless]
                body_angle  [Radians]
            
        Outputs:
            state.conditions:
                propulsion.throttle            [Unitless]
                frames.body.inertial_rotations [Radians]

        Properties Used:
        N/A
                                
    """    
    
    # unpack unknowns
    throttle   = state.unknowns.throttle
    body_angle = state.unknowns.body_angle
    
    # apply unknowns
    state.conditions.propulsion.throttle[:,0]            = throttle[:,0]
    state.conditions.frames.body.inertial_rotations[:,1] = body_angle[:,0]   
    

# ----------------------------------------------------------------------
#  Residual Total Forces
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Cruise
def residual_total_forces(segment,state):
    """ Calculates a residual based on forces
    
        Assumptions:
        The vehicle is not accelerating, doesn't use gravity
        
        Inputs:
            state.conditions:
                frames.inertial.total_force_vector [Newtons]
                weights.total_mass                 [kg]
            
        Outputs:
            state.residuals.forces [meters/second^2]

        Properties Used:
        N/A
                                
    """        
    
    FT = state.conditions.frames.inertial.total_force_vector
    m  = state.conditions.weights.total_mass[:,0] 
    
    # horizontal
    state.residuals.forces[:,0] = np.sqrt( FT[:,0]**2. + FT[:,1]**2. )/m
    # vertical
    state.residuals.forces[:,1] = FT[:,2]/m

    return
    
    
 
    
    