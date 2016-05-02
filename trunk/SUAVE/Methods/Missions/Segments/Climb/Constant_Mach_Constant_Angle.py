# Constant_Mach_Constant_Angle.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero

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
    # unpack user inputs
    climb_angle = segment.climb_angle
    mach_number = segment.mach
    alt0        = segment.altitude_start 
    altf        = segment.altitude_end
    t_nondim    = state.numerics.dimensionless.control_points
    conditions  = state.conditions 
 
    # Update freestream to get speed of sound
    SUAVE.Methods.Missions.Segments.Common.Aerodynamics.update_atmosphere(segment,state)
    a = conditions.freestream.speed_of_sound
    # check for initial altitude
    if alt0 is None:
        if not state.initials: raise AttributeError('initial altitude not set')
        alt0 = -1.0 * state.initials.conditions.frames.inertial.position_vector[-1,2]

    # discretize on altitude
    alt = t_nondim * (altf-alt0) + alt0
    
    # process velocity vector
    v_mag = mach_number * a
    v_x   = v_mag * np.cos(climb_angle)
    v_z   = -v_mag * np.sin(climb_angle)
    
    # pack conditions    
    conditions.frames.inertial.velocity_vector[:,0] = v_x[:,0]
    conditions.frames.inertial.velocity_vector[:,2] = v_z[:,0]
    conditions.frames.inertial.position_vector[:,2] = -alt[:,0] # z points down
    conditions.freestream.altitude[:,0]             =  alt[:,0] # positive altitude in this context
