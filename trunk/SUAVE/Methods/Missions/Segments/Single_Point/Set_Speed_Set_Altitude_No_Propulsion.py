## @ingroup Methods-Missions-Segments-Single_Point
# Set_Speed_Set_Altitude.py
# 
# Created:  Mar 2017, T. MacDonald
# Modified: Jul 2017, T. MacDonald
#           Aug 2017, E. Botero
#           May 2019, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Single_Point
def initialize_conditions(segment):
    """Sets the specified conditions which are given for the segment type.

    Assumptions:
    A fixed speed and altitude

    Source:
    N/A

    Inputs:
    segment.altitude                            [meters]
    segment.air_speed                           [meters/second]
    segment.z_accel                             [meters/second^2]

    Outputs:
    conditions.frames.inertial.acceleration_vector [meters/second^2]
    conditions.frames.inertial.velocity_vector     [meters/second]
    conditions.frames.inertial.position_vector     [meters]
    conditions.freestream.altitude                 [meters]
    conditions.frames.inertial.time                [seconds]

    Properties Used:
    N/A
    """      
    
    # unpack
    alt        = segment.altitude
    air_speed  = segment.air_speed  
    z_accel    = segment.z_accel
    conditions = segment.state.conditions 
    
    # check for initial altitude
    if alt is None:
        if not segment.state.initials: raise AttributeError('altitude not set')
        alt = -1.0 *segment.state.initials.conditions.frames.inertial.position_vector[-1,2]
    
    # pack
    segment.state.conditions.freestream.altitude[:,0]             = alt
    segment.state.conditions.frames.inertial.position_vector[:,2] = -alt # z points down
    segment.state.conditions.frames.inertial.velocity_vector[:,0] = air_speed
    segment.state.conditions.frames.inertial.acceleration_vector  = np.array([[0.0,0.0,z_accel]])
    
    
## @ingroup Methods-Missions-Segments-Cruise
def unpack_unknowns(segment):
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
    body_angle = segment.state.unknowns.body_angle

    # apply unknowns
    segment.state.conditions.frames.body.inertial_rotations[:,1] = body_angle[:,0]   
        
    
    
    
# ----------------------------------------------------------------------
#  Residual Total Forces
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Climb
def residual_total_force(segment):
    """Takes the summation of forces and makes a residual from the accelerations.

    Assumptions:
    No higher order terms.

    Source:
    N/A

    Inputs:
    segment.state.conditions.frames.inertial.total_force_vector   [Newtons]
    segment.state.conditions.frames.inertial.acceleration_vector  [meter/second^2]
    segment.state.conditions.weights.total_mass                   [kilogram]

    Outputs:
    segment.state.residuals.force                                 [Unitless]

    Properties Used:
    N/A
    """        
    
    FT = segment.state.conditions.frames.inertial.total_force_vector
    a  = segment.state.conditions.frames.inertial.acceleration_vector
    m  = segment.state.conditions.weights.total_mass    
    
    segment.state.residuals.force = FT[:,2]/m[:,0] - a[:,2]

    return    
