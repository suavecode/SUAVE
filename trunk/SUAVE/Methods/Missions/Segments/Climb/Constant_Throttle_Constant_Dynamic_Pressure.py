import numpy as np
import SUAVE

# ----------------------------------------------------------------------
#  Unpack Unknowns
# ----------------------------------------------------------------------

def unpack_body_angle(segment,state):

    # unpack unknowns
    theta      = state.unknowns.body_angle

    # apply unknowns
    state.conditions.frames.body.inertial_rotations[:,1] = theta[:,0]      


# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------


def initialize_conditions(segment,state):
    
    # unpack
    throttle   = segment.throttle
    q          = segment.dynamic_pressure
    alt0       = segment.altitude_start 
    altf       = segment.altitude_end
    t_nondim   = state.numerics.dimensionless.control_points
    conditions = state.conditions  
    
    # Update freestream to get density
    SUAVE.Methods.Missions.Segments.Common.Aerodynamics.update_atmosphere(segment,state)
    rho        = conditions.freestream.density[:,0]    
    
    air_speed  = np.sqrt(q/rho)

    # check for initial altitude
    if alt0 is None:
        if not state.initials: raise AttributeError('initial altitude not set')
        alt0 = -1.0 * state.initials.conditions.frames.inertial.position_vector[-1,2]
        segment.altitude_start = alt0

    # discretize on altitude
    alt = t_nondim * (altf-alt0) + alt0
    
    # pack conditions  
    conditions.propulsion.throttle[:,0] = throttle
    conditions.frames.inertial.velocity_vector[:,0] = air_speed # start up value
    conditions.frames.inertial.position_vector[:,2] = -alt[:,0] # z points down
    conditions.freestream.altitude[:,0]             =  alt[:,0] # positive altitude in this context
    
    
def update_velocity_vector_from_wind_angle(segment,state):
    
    # unpack
    conditions = state.conditions 
    q          = segment.dynamic_pressure
    theta      = state.unknowns.wind_angle[:,0][:,None]
    
    # Update freestream to get density
    SUAVE.Methods.Missions.Segments.Common.Aerodynamics.update_atmosphere(segment,state)
    rho        = conditions.freestream.density[:,0]    
    
    v_mag  = np.sqrt(q/rho)    

    # process
    v_x =  v_mag * np.cos(theta)
    v_z = -v_mag * np.sin(theta) # z points down

    # pack
    conditions.frames.inertial.velocity_vector[:,0] = v_x[:,0]
    conditions.frames.inertial.velocity_vector[:,2] = v_z[:,0]

    return conditions    
