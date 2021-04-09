## @ingroup Methods-Missions-Segments-Hover
# Hover.py
# 
# Created:  Jan 2016, E. Botero
# Modified: May 2019, T. MacDonald
#           Mar 2020, M. Clarke

# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Hover
def initialize_conditions(segment):
    """Sets the specified conditions which are given for the segment type.

    Assumptions:
    Descent segment with a constant rate.

    Source:
    N/A

    Inputs:
    segment.altitude                            [meters]
    segment.tim                                 [second]
    state.numerics.dimensionless.control_points [Unitless]
    state.conditions.frames.inertial.time       [seconds]

    Outputs:
    conditions.frames.inertial.velocity_vector  [meters/second]
    conditions.frames.inertial.position_vector  [meters]
    conditions.freestream.altitude              [meters]
    conditions.frames.inertial.time             [seconds]

    Properties Used:
    N/A
    """       
    
    # unpack
    alt        = segment.altitude
    duration   = segment.time
    conditions = segment.state.conditions   
    
    
    # check for initial altitude
    if alt is None:
        if not segment.state.initials: raise AttributeError('altitude not set')
        alt = -1.0 *segment.state.initials.conditions.frames.inertial.position_vector[-1,2]      
    
    # dimensionalize time
    t_initial = conditions.frames.inertial.time[0,0]
    t_nondim  = segment.state.numerics.dimensionless.control_points
    time      =  t_nondim * (duration) + t_initial
    
    # pack
    segment.state.conditions.freestream.altitude[:,0]             = alt
    segment.state.conditions.frames.inertial.position_vector[:,2] = -alt # z points down
    segment.state.conditions.frames.inertial.velocity_vector[:,0] = 0.
    segment.state.conditions.frames.inertial.time[:,0]            = time[:,0]    
