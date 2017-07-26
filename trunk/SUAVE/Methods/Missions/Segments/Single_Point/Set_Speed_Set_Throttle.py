# Set_Speed_Set_Throttle.py
# 
# Created:  Mar 2017, T. MacDonald
# Modified: Jun 2017, T. MacDonald

# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------

def initialize_conditions(segment,state):
    
    # unpack
    alt        = segment.altitude
    air_speed  = segment.air_speed
    throttle   = segment.throttle
    conditions = state.conditions 
    
    # check for initial altitude
    if alt is None:
        if not state.initials: raise AttributeError('altitude not set')
        alt = -1.0 * state.initials.conditions.frames.inertial.position_vector[-1,2]
        segment.altitude = alt
    
    # pack
    state.conditions.freestream.altitude[:,0]             = alt
    state.conditions.frames.inertial.position_vector[:,2] = -alt # z points down
    state.conditions.frames.inertial.velocity_vector[:,0] = air_speed
    state.conditions.propulsion.throttle[:,0]             = throttle
    
def update_weights(segment,state):
    
    # unpack
    conditions = state.conditions
    m0         = conditions.weights.total_mass[0,0]
    g          = conditions.freestream.gravity

    # weight
    W = m0*g

    # pack
    conditions.frames.inertial.gravity_force_vector[:,2] = W

    return

def unpack_unknowns(segment,state):
    
    # unpack unknowns
    x_accel    = state.unknowns.x_accel
    body_angle = state.unknowns.body_angle
    
    # apply unknowns
    state.conditions.frames.inertial.acceleration_vector[0,0] = x_accel
    state.conditions.frames.body.inertial_rotations[:,1] = body_angle[:,0]      