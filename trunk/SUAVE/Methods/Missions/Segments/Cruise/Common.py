


# ----------------------------------------------------------------------
#  Unpack Unknowns
# ----------------------------------------------------------------------

def initialize_conditions(segment,state):
    
    # unpack
    alt       = segment.altitude
    xf        = segment.distance
    air_speed = segment.air_speed   
    
    conditions = state.conditions    
    t_initial = conditions.frames.inertial.time[0,0]
    
    t_nondim  = state.numerics.dimensionless.control_points
    
    # pack
    conditions.freestream.altitude[:,0] = alt
    conditions.frames.inertial.position_vector[:,2] = -alt # z points down
    conditions.frames.inertial.velocity_vector[:,0] = air_speed
    
    # dimensionalize time
    t_final = xf / air_speed + t_initial
    time =  t_nondim * (t_final-t_initial) + t_initial
    conditions.frames.inertial.time[:,0] = time[:,0]
    

# ----------------------------------------------------------------------
#  Unpack Unknowns
# ----------------------------------------------------------------------

def unpack_unknowns(segment,state):
    
    # unpack unknowns
    throttle   = state.unknowns.throttle
    body_angle = state.unknowns.body_angle
    
    # apply unknowns
    state.conditions.propulsion.throttle[:,0]            = throttle[:,0]
    state.conditions.frames.body.inertial_rotations[:,1] = body_angle[:,0]   
    

# ----------------------------------------------------------------------
#  Residual Total Forces
# ----------------------------------------------------------------------

def residual_total_forces(segment,state):
    
    FT = state.conditions.frames.inertial.total_force_vector
    
    state.residuals.forces[:,0] = FT[:,0]
    state.residuals.forces[:,1] = FT[:,2]    

    return
    
    
 
    
    