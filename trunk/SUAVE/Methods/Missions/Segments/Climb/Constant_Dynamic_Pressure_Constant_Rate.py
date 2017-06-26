# Constant_Dynamic_Pressure_Constant_Rate.py
# 
# Created:  Jan 2016, E. Botero
# Modified: Jun 2017, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np
import SUAVE

# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------

def initialize_conditions(segment,state):
    
    # unpack
    climb_rate = segment.climb_rate
    q          = segment.dynamic_pressure
    alt0       = segment.altitude_start 
    altf       = segment.altitude_end
    t_nondim   = state.numerics.dimensionless.control_points
    conditions = state.conditions
    rho        = conditions.freestream.density[:,0]
    
    # Update freestream to get density
    SUAVE.Methods.Missions.Segments.Common.Aerodynamics.update_atmosphere(segment,state)
    rho = conditions.freestream.density[:,0]   

    # check for initial altitude
    if alt0 is None:
        if not state.initials: raise AttributeError('initial altitude not set')
        alt0 = -1.0 * state.initials.conditions.frames.inertial.position_vector[-1,2]

    # discretize on altitude
    alt = t_nondim * (altf-alt0) + alt0
    
    # process velocity vector
    v_mag = np.sqrt(2*q/rho)
    v_z   = -climb_rate # z points down
    v_x   = np.sqrt( v_mag**2 - v_z**2 )
    
    # pack conditions    
    conditions.frames.inertial.velocity_vector[:,0] = v_x
    conditions.frames.inertial.velocity_vector[:,2] = v_z
    conditions.frames.inertial.position_vector[:,2] = -alt[:,0] # z points down
    conditions.freestream.altitude[:,0]             =  alt[:,0] # positive altitude in this context