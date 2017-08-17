## @ingroup Methods-Missions-Segments-Climb
# Constant_Speed_Constant_Rate.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import autograd.numpy as np 

# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------
## @ingroup Methods-Missions-Segments-Climb
def initialize_conditions(segment,state):
    
    """Sets the specified conditions which are given for the segment type.
    
    Assumptions:
    Constant true airspeed, with a constant rate of climb

    Source:
    N/A

    Inputs:
    segment.climb_rate                          [meters/second]
    segment.air_speed                           [meters/second]
    segment.altitude_start                      [meters]
    segment.altitude_end                        [meters]
    state.numerics.dimensionless.control_points [Unitless]
    conditions.freestream.density               [kilograms/meter^3]

    Outputs:
    conditions.frames.inertial.velocity_vector  [meters/second]
    conditions.frames.inertial.position_vector  [meters]
    conditions.freestream.altitude              [meters]

    Properties Used:
    N/A
    """            
    
    # unpack
    climb_rate = segment.climb_rate
    air_speed  = segment.air_speed   
    alt0       = segment.altitude_start 
    altf       = segment.altitude_end
    t_nondim   = state.numerics.dimensionless.control_points
    conditions = state.conditions  

    # check for initial altitude
    if alt0 is None:
        if not state.initials: raise AttributeError('initial altitude not set')
        alt0 = -1.0 * state.initials.conditions.frames.inertial.position_vector[-1,2]

    # discretize on altitude
    alt = t_nondim * (altf-alt0) + alt0
    
    # process velocity vector
    v_mag = air_speed
    v_z   = -climb_rate # z points down
    v_x   = np.sqrt( v_mag**2 - v_z**2 )
    
    cond = state.conditions
    v_xs = state.ones_row(1) * v_x
    v_zs = state.ones_row(1) * v_z
    zero = state.ones_row(1) * 0.
    
    # pack conditions    
    cond.frames.inertial.velocity_vector = np.concatenate((v_xs,zero,v_zs),axis=1)
    cond.frames.inertial.position_vector = np.concatenate((cond.frames.inertial.position_vector[:,0:2],-alt),axis=1) # z points down
    cond.freestream.altitude             = alt # positive altitude in this context