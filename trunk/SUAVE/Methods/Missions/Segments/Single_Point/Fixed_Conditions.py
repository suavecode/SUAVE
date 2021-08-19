## @ingroup Methods-Missions-Segments-Single_Point
# Fixed_Conditions.py
# 
# Created:  Aug 2021, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Single_Point
def initialize_conditions(segment):
    """Sets the specified conditions which are given for the segment type.

    Assumptions:
       A single-point analysis for a fixed set of parameters including: speed, throttle,
       body angle, and altitude. 

    Source:
       N/A

    Inputs:
       segment.altitude                               [meters]
       segment.air_speed                              [meters/second]
       segment.throttle                               [unitless]
       segment.z_accel                                [meters/second^2]
       segment.state.unknowns.x_accel                 [meters/second^2]

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
    throttle   = segment.throttle
    body_angle = segment.body_angle
    z_accel    = segment.state.unknowns.z_accel
    x_accel    = segment.state.unknowns.x_accel
    conditions = segment.state.conditions 
    
    # check for initial altitude
    if alt is None:
        if not segment.state.initials: raise AttributeError('altitude not set')
        alt = -1.0 *segment.state.initials.conditions.frames.inertial.position_vector[-1,2]
    
    # pack
    conditions.freestream.altitude[:,0]                  = alt.T
    conditions.frames.inertial.position_vector[:,2]      = -alt.T # z points down
    conditions.frames.inertial.velocity_vector[:,0]      = air_speed.T
    conditions.frames.body.inertial_rotations[:,1]       = body_angle.T
    conditions.propulsion.throttle[:,0]                  = throttle.T
    conditions.frames.inertial.acceleration_vector[:,0]  = x_accel.T
    conditions.frames.inertial.acceleration_vector[:,2]  = z_accel.T

## @ingroup Methods-Missions-Segments-Single_Point    
def update_weights(segment):
    """Sets the gravity force vector during the segment

    Assumptions:
    A fixed speed and altitde

    Source:
    N/A

    Inputs:
    conditions:
        weights.total_mass                          [kilogram]
        freestream.gravity                          [meters/second^2]

    Outputs:
    conditions.frames.inertial.gravity_force_vector [newtons]


    Properties Used:
    N/A
    """         
    
    # unpack
    conditions = segment.state.conditions
    m0         = conditions.weights.total_mass[0,0]
    g          = conditions.freestream.gravity

    # weight
    W = m0*g

    # pack
    conditions.frames.inertial.gravity_force_vector[:,2] = W.T[0]

    return

## @ingroup Methods-Missions-Segments-Single_Point
def unpack_unknowns(segment):
    """ Unpacks the x accleration and body angle from the solver to the mission
    
        Assumptions:
        N/A
        
        Inputs:
            segment.state.unknowns:
                x_accel                             [meters/second^2]
                body_angle                          [radians]
            
        Outputs:
            segment.state.conditions:
                frames.inertial.acceleration_vector [meters/second^2]
                frames.body.inertial_rotations      [radians]

        Properties Used:
        N/A
                                
    """      
    
    # unpack unknowns
    x_accel    = segment.state.unknowns.x_accel
    z_accel    = segment.state.unknowns.z_accel
    
    # apply unknowns
    segment.state.conditions.frames.inertial.acceleration_vector[:,0] = x_accel.T
    segment.state.conditions.frames.inertial.acceleration_vector[:,2] = z_accel.T     