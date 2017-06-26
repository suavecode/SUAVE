# Constant_Dynamic_Pressure_Constant_Angle.py
# 
# Created:  Jun 2017, E. Botero
# Modified:          

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np
import SUAVE

# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------

def initialize_conditions_unpack_unknowns(segment,state):
    
    # unpack
    climb_angle = segment.climb_angle
    q           = segment.dynamic_pressure
    alt0        = segment.altitude_start 
    altf        = segment.altitude_end
    t_nondim    = state.numerics.dimensionless.control_points
    conditions  = state.conditions
    rho         = conditions.freestream.density[:,0]
    
    # unpack unknowns
    throttle = state.unknowns.throttle
    theta    = state.unknowns.body_angle
    alts     = state.unknowns.altitudes    

    # Update freestream to get density
    SUAVE.Methods.Missions.Segments.Common.Aerodynamics.update_atmosphere(segment,state)
    rho = conditions.freestream.density[:,0]   

    # check for initial altitude
    if alt0 is None:
        if not state.initials: raise AttributeError('initial altitude not set')
        alt0 = -1.0 * state.initials.conditions.frames.inertial.position_vector[-1,2]
    
    # pack conditions    
    conditions.freestream.altitude[:,0]             =  alts[:,0] # positive altitude in this context    
    
    # Update freestream to get density
    SUAVE.Methods.Missions.Segments.Common.Aerodynamics.update_atmosphere(segment,state)
    rho = conditions.freestream.density[:,0]       
    
    # process velocity vector
    v_mag = np.sqrt(2*q/rho)
    v_x   = v_mag * np.cos(climb_angle)
    v_z   = -v_mag * np.sin(climb_angle)
    
    # pack conditions    
    conditions.frames.inertial.velocity_vector[:,0] = v_x
    conditions.frames.inertial.velocity_vector[:,2] = v_z
    conditions.frames.inertial.position_vector[:,2] = -alts[:,0] # z points down
    state.conditions.propulsion.throttle[:,0]            = throttle[:,0]
    state.conditions.frames.body.inertial_rotations[:,1] = theta[:,0]  
    

def residual_total_forces(segment,state):
    
    FT = state.conditions.frames.inertial.total_force_vector
    a  = state.conditions.frames.inertial.acceleration_vector
    m  = state.conditions.weights.total_mass    
    alt_in  = state.unknowns.altitudes[:,0] 
    alt_out = state.conditions.freestream.altitude[:,0] 
    
    state.residuals.forces[:,0] = FT[:,0]/m[:,0] - a[:,0]
    state.residuals.forces[:,1] = FT[:,2]/m[:,0] - a[:,2]
    state.residuals.forces[:,2] = (alt_in - alt_out)/alt_out[-1]

    return