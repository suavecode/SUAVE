
import numpy as np

# ----------------------------------------------------------------------
#  Unpack Unknowns
# ----------------------------------------------------------------------

def initialize_conditions(segment,state):
    
    # unpack
    alt        = segment.altitude 
    T0         = segment.pitch_initial
    Tf         = segment.pitch_final 
    theta_dot  = segment.pitch_rate   
    conditions = state.conditions 
    
    # check for initial altitude
    if alt is None:
        if not state.initials: raise AttributeError('altitude not set')
        alt = -1.0 * state.initials.conditions.frames.inertial.position_vector[-1,2]
        T0  =  state.initials.conditions.frames.body.inertial_rotations[-1,1]
        segment.altitude = alt
        segment.pitch_initial = T0
    
    # dimensionalize time
    t_initial = conditions.frames.inertial.time[0,0]
    t_final   = (Tf-T0)/theta_dot + t_initial
    t_nondim  = state.numerics.dimensionless.control_points
    time      = t_nondim * (t_final-t_initial) + t_initial
    
    # set the body angle
    body_angle = theta_dot*time + T0
    state.conditions.frames.body.inertial_rotations[:,1] = body_angle[:,0]    
    
    # pack
    state.conditions.freestream.altitude[:,0]             = alt
    state.conditions.frames.inertial.position_vector[:,2] = -alt # z points down
    state.conditions.frames.inertial.time[:,0]            = time[:,0]
    
    
    
def residual_total_forces(segment,state):
    
    FT = state.conditions.frames.inertial.total_force_vector
    m  = state.conditions.weights.total_mass  
    v  = state.conditions.frames.inertial.velocity_vector
    D  = state.numerics.time.differentiate
    m  = state.conditions.weights.total_mass    
    
    # process and pack
    acceleration = np.dot(D,v)
    state.conditions.frames.inertial.acceleration_vector = acceleration
    a  = state.conditions.frames.inertial.acceleration_vector
    
    # horizontal
    state.residuals.forces[:,0] = FT[:,0]/m[:,0] - a[:,0]
    # vertical
    state.residuals.forces[:,1] = FT[:,2]  - a[:,2]

    return

def unpack_unknowns(segment,state):
    
    # unpack unknowns
    throttle  = state.unknowns.throttle
    air_speed = state.unknowns.velocity
    
    # apply unknowns
    state.conditions.propulsion.throttle[:,0]             = throttle[:,0]
    state.conditions.frames.inertial.velocity_vector[:,0] = air_speed[:,0]
    
    