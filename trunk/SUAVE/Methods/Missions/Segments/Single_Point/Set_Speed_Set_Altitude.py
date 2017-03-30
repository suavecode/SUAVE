# Set_Speed_Set_Altitude.py
# 
# Created:  Mar 2017, T. MacDonald
# Modified: 

# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------

def initialize_conditions(segment,state):
    
    # unpack
    alt        = segment.altitude
    air_speed  = segment.air_speed       
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