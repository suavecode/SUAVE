## @ingroup Methods-Missions-Segments-Hover
# Climb.py
# 
# Created:  Jan 2016, E. Botero
# Modified:

# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Hover
def initialize_conditions(segment):
    """Sets the specified conditions which are given for the segment type.

    Assumptions:
    Climb segment with a constant rate of climb.

    Source:
    N/A

    Inputs:
    segment.altitude_start                              [meters]
    segment.altitude_end                                [meters]
    segment.climb_rate                                  [meters/second]
    segment.state.numerics.dimensionless.control_points [Unitless]
    segment.state.conditions.frames.inertial.time       [seconds]

    Outputs:
    conditions.frames.inertial.velocity_vector  [meters/second]
    conditions.frames.inertial.position_vector  [meters]
    conditions.freestream.altitude              [meters]
    conditions.frames.inertial.time             [seconds]

    Properties Used:
    N/A
    """       
    
    # unpack
    climb_rate = segment.climb_rate
    alt0       = segment.altitude_start 
    altf       = segment.altitude_end
    t_nondim   = segment.state.numerics.dimensionless.control_points
    t_initial  = segment.state.conditions.frames.inertial.time[0,0]
    conditions = segment.state.conditions  

    # check for initial altitude
    if alt0 is None:
        if not segment.state.initials: raise AttributeError('initial altitude not set')
        alt0 = -1.0 * segment.state.initials.conditions.frames.inertial.position_vector[-1,2]

    # discretize on altitude
    alt = t_nondim * (altf-alt0) + alt0
    
    # process velocity vector
    v_z   = -climb_rate # z points down    
    dt = (altf - alt0)/climb_rate

    # rescale operators
    t = t_nondim * dt

    # pack
    t_initial = segment.state.conditions.frames.inertial.time[0,0]
    segment.state.conditions.frames.inertial.time[:,0] = t_initial + t[:,0]    
    
    # pack conditions    
    conditions.frames.inertial.velocity_vector[:,0] = 0.
    conditions.frames.inertial.velocity_vector[:,2] = v_z
    conditions.frames.inertial.position_vector[:,2] = -alt[:,0] # z points down
    conditions.freestream.altitude[:,0]             =  alt[:,0] # positive altitude in this context
    segment.state.conditions.frames.inertial.time[:,0]      = t_initial + t[:,0]