
# ----------------------------------------------------------------------
#  Unpack Unknowns
# ----------------------------------------------------------------------

def unpack_unknowns(segment,state):
    
    # unpack unknowns
    throttle   = state.unknowns.throttle
    
    # apply unknowns
    state.conditions.propulsion.throttle[:,0] = throttle[:,0]
    

# ----------------------------------------------------------------------
#  Residual Total Forces
# ----------------------------------------------------------------------

def residual_total_forces(segment,state):
    
    FT = state.conditions.frames.inertial.total_force_vector

    # vertical
    state.residuals.forces[:,0] = FT[:,2]

    return
    
    
 
    
    