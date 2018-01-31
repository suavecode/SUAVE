## @ingroup Methods-Missions-Segments-Climb
# Constant_Speed_Linear_Altitude.py 
# Created:  Jul 2014, SUAVE Team
# Modified: Jun 2017, E. Botero

# ----------------------------------------------------------------------
#  initialize conditions
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Climb
def initialize_conditions(segment,state):
    """Sets the specified conditions which are given for the segment type.

    Assumptions:
    Constrant dynamic pressure and constant rate of climb

    Source:
    N/A

    Inputs:
    segment.air_speed                           [meters/second]
    segment.altitude_start                      [meters]
    segment.altitude_end                        [meters]
    segment.distance                            [meters]

    Outputs:
    conditions.frames.inertial.velocity_vector  [meters/second]
    conditions.frames.inertial.position_vector  [meters]
    conditions.freestream.altitude              [meters]
    conditions.frames.inertial.time             [seconds]

    Properties Used:
    N/A
    """        
    
    # unpack
    alt0       = segment.altitude_start 
    altf       = segment.altitude_end
    xf         = segment.distance
    air_speed  = segment.air_speed       
    conditions = state.conditions 
    
    # check for initial altitude
    if alt0 is None:
        if not state.initials: raise AttributeError('altitude not set')
        alt0 = -1.0 * state.initials.conditions.frames.inertial.position_vector[-1,2]
        #segment.altitude = alt
    
    # dimensionalize time
    t_initial = conditions.frames.inertial.time[0,0]
    t_final   = xf / air_speed + t_initial
    t_nondim  = state.numerics.dimensionless.control_points
    time      = t_nondim * (t_final-t_initial) + t_initial
    
    # discretize on altitude
    alt = t_nondim * (altf-alt0) + alt0    
    
    segment.altitude = 0.5*(alt0 + altf)
    
    # pack
    state.conditions.freestream.altitude[:,0] = alt[:,0]
    state.conditions.frames.inertial.position_vector[:,2] = -alt[:,0] # z points down
    state.conditions.frames.inertial.velocity_vector[:,0] = air_speed
    state.conditions.frames.inertial.time[:,0] = time[:,0]