## @ingroup Methods-Missions-Segments-Climb
# Linear_Speed_Constant_Rate.py
# 
# Created:  Jul 2014, SUAVE Team (Stanford University)
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Climb
def initialize_conditions(segment):
    """Sets the specified conditions which are given for the segment type.
    
    Assumptions:
    Linearly changing airspeed, with a constant rate of climb

    Source:
    N/A

    Inputs:
    segment.climb_rate                          [meters/second]
    segment.air_speed_start                     [meters/second]
    segment.air_speed_end                       [meters/second]
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
    # unpack user inputs
    climb_rate = segment.climb_rate
    Vo         = segment.air_speed_start
    Vf         = segment.air_speed_end
    alt0       = segment.altitude_start 
    altf       = segment.altitude_end
    t_nondim   = segment.state.numerics.dimensionless.control_points
    conditions = segment.state.conditions  

    # check for initial altitude
    if alt0 is None:
        if not segment.state.initials: raise AttributeError('initial altitude not set')
        alt0 = -1.0 *segment.state.initials.conditions.frames.inertial.position_vector[-1,2]

    # discretize on altitude
    alt = t_nondim * (altf-alt0) + alt0
    
    # process velocity vector
    v_mag = (Vf-Vo)*t_nondim + Vo
    v_z   = -climb_rate # z points down
    v_x   = np.sqrt( v_mag**2 - v_z**2 )
    
    # pack conditions    
    conditions.frames.inertial.velocity_vector[:,0] = v_x[:,0]
    conditions.frames.inertial.velocity_vector[:,2] = v_z
    conditions.frames.inertial.position_vector[:,2] = -alt[:,0] # z points down
    conditions.freestream.altitude[:,0]             =  alt[:,0] # positive altitude in this context