


# ----------------------------------------------------------------------
#  Unpack Unknowns
# ----------------------------------------------------------------------

def initialize_conditions(segment,state):
    
    # unpack
    alt       = segment.altitude
    xf        = segment.distance
    air_speed = segment.air_speed   
    
    # check for initial altitude
    if alt is None:
        if not state.initials: raise AttributeError('altitude not set')
        alt = -1.0 * state.initials.conditions.frames.inertial.position_vector[-1,2]
        segment.altitude = alt    
    
    conditions = state.conditions    
    t_initial = conditions.frames.inertial.time[0,0]
    t_final   = xf / air_speed + t_initial
    t_nondim  = state.numerics.dimensionless.control_points
    
    # pack
    state.conditions.freestream.altitude[:,0] = alt
    state.conditions.frames.inertial.position_vector[:,2] = -alt # z points down
    state.conditions.frames.inertial.velocity_vector[:,0] = air_speed
    
    # dimensionalize time
    time =  t_nondim * (t_final-t_initial) + t_initial
    state.conditions.frames.inertial.time[:,0] = time[:,0]