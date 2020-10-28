## @ingroup Methods-Missions-Segments-Climb
# Constant_Mach_Linear_Altitude.py

# Created:  Jul 2014, SUAVE Team
# Modified: Jun 2017, E. Botero 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE

# ----------------------------------------------------------------------
#  initialize conditions
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Climb
def initialize_conditions(segment):
    """Sets the specified conditions which are given for the segment type.

    Assumptions:
    Constrant dynamic pressure and constant rate of climb

    Source:
    N/A

    Inputs:
    segment.mach                                [unitless]
    segment.dynamic_pressure                    [pascals]
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
    mach       = segment.mach
    conditions = segment.state.conditions 
    t_initial  = conditions.frames.inertial.time[0,0]    
    t_nondim   = segment.state.numerics.dimensionless.control_points
    
    # check for initial altitude
    if alt0 is None:
        if not segment.state.initials: raise AttributeError('altitude not set')
        alt0 = -1.0 * segment.state.initials.conditions.frames.inertial.position_vector[-1,2]
        
    # discretize on altitude
    alt = t_nondim * (altf-alt0) + alt0      
    segment.state.conditions.freestream.altitude[:,0] = alt[:,0]
        
    # Update freestream to get speed of sound
    SUAVE.Methods.Missions.Segments.Common.Aerodynamics.update_atmosphere(segment)
    a          = conditions.freestream.speed_of_sound    

    # compute speed, constant with constant altitude
    air_speed = mach * a
    
    # dimensionalize time
    t_final   = xf / air_speed + t_initial 
    time      = t_nondim * (t_final-t_initial) + t_initial
    
    segment.altitude = 0.5*(alt0 + altf)
    
    # pack
    segment.state.conditions.freestream.altitude[:,0]             = alt[:,0]
    segment.state.conditions.frames.inertial.position_vector[:,2] = -alt[:,0] # z points down
    segment.state.conditions.frames.inertial.velocity_vector[:,0] = air_speed[:,0]
    segment.state.conditions.frames.inertial.time[:,0]            = time[:,0]