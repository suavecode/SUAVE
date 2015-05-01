


# ----------------------------------------------------------------------
#  Unpack Unknowns
# ----------------------------------------------------------------------

def initialize_conditions(segment,state):
    
    # unpack
    alt        = segment.altitude
    xf         = segment.distance
    mach       = segment.mach
    atmo       = segment.atmo
    conditions = state.conditions    
    
    # check for initial altitude
    if alt is None:
        if not state.initials: raise AttributeError('altitude not set')
        alt = -1.0 * state.initials.conditions.frames.inertial.position_vector[-1,2]
        segment.altitude = alt        
    
    # compute speed, constant with constant altitude
    atmo_cond = atmo.compute_values(alt)
    air_speed = mach * atmo_cond.speed_of_sound  
    
    # dimensionalize time
    t_initial = conditions.frames.inertial.time[0,0]
    t_final   = xf / air_speed + t_initial
    t_nondim  = state.numerics.dimensionless.control_points
    time      =  t_nondim * (t_final-t_initial) + t_initial
    
    # pack
    state.conditions.freestream.altitude[:,0] = alt
    state.conditions.frames.inertial.position_vector[:,2] = -alt # z points down
    state.conditions.frames.inertial.velocity_vector[:,0] = air_speed
    state.conditions.frames.inertial.time[:,0] = time[:,0]