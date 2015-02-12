
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
    
    
 
    
    