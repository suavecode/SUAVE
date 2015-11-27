import SUAVE
import numpy as np

# ----------------------------------------------------------------------
#  Unpack Unknowns
# ----------------------------------------------------------------------

def initialize_conditions(segment,state):
    
    # unpack
    alt        = segment.altitude
    final_time = segment.time
    q          = segment.dynamic_pressure
    conditions = state.conditions   
    
    # Update freestream to get density
    SUAVE.Methods.Missions.Segments.Common.Aerodynamics.update_atmosphere(segment,state)
    rho        = conditions.freestream.density[:,0]   
    
    # check for initial altitude
    if alt is None:
        if not state.initials: raise AttributeError('altitude not set')
        alt = -1.0 * state.initials.conditions.frames.inertial.position_vector[-1,2]
        segment.altitude = alt        
    
    # compute speed, constant with constant altitude
    air_speed = np.sqrt(q/(rho*0.5))
    
    # dimensionalize time
    t_initial = conditions.frames.inertial.time[0,0]
    t_final   = final_time + t_initial
    t_nondim  = state.numerics.dimensionless.control_points
    time      =  t_nondim * (t_final-t_initial) + t_initial
    
    # pack
    state.conditions.freestream.altitude[:,0]             = alt
    state.conditions.frames.inertial.position_vector[:,2] = -alt # z points down
    state.conditions.frames.inertial.velocity_vector[:,0] = air_speed
    state.conditions.frames.inertial.time[:,0] = time[:,0]
    
