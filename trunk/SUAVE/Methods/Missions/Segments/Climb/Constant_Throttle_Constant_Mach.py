## @ingroup Methods-Missions-Segments-Climb
# Constant_Throttle_Constant_Mach.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
import SUAVE

# ----------------------------------------------------------------------
#  Unpack Unknowns
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Climb
def initialize_conditions(segment):
    """Sets the specified conditions which are given for the segment type.
    
    Assumptions:
    Constant throttle estting, with a constant true airspeed

    Source:
    N/A

    Inputs:
    segment.air_speed                                   [meters/second]
    segment.throttle                                    [Unitless]
    segment.altitude_start                              [meters]
    segment.altitude_end                                [meters]
    segment.state.numerics.dimensionless.control_points [Unitless]
    conditions.freestream.density                       [kilograms/meter^3]

    Outputs:
    conditions.frames.inertial.velocity_vector  [meters/second]
    conditions.propulsion.throttle              [Unitless]

    Properties Used:
    N/A
    """         
    
    # unpack
    throttle    = segment.throttle
    mach        = segment.mach
    alt0        = segment.altitude_start 
    altf        = segment.altitude_end
    t_nondim    = segment.state.numerics.dimensionless.control_points
    conditions  = segment.state.conditions  

    # check for initial altitude
    if alt0 is None:
        if not segment.state.initials: raise AttributeError('initial altitude not set')
        alt0 = -1.0 *segment.state.initials.conditions.frames.inertial.position_vector[-1,2]

    # pack conditions  
    # Update freestream to get speed of sound
    SUAVE.Methods.Missions.Segments.Common.Aerodynamics.update_atmosphere(segment)
    a         = conditions.freestream.speed_of_sound
    air_speed = mach * a
    
    conditions.propulsion.throttle[:,0] = throttle
    conditions.frames.inertial.velocity_vector[:,0] = air_speed[:,0] # start up value

# ----------------------------------------------------------------------
#  Update Velocity Vector from Wind Angle
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Climb
def update_velocity_vector_from_wind_angle(segment):
    
    # unpack
    conditions = segment.state.conditions 
    mach       = segment.mach
    SUAVE.Methods.Missions.Segments.Common.Aerodynamics.update_atmosphere(segment)
    a         = conditions.freestream.speed_of_sound
    air_speed = mach * a
    
    # process velocity vector
    v_mag = air_speed
    #v_mag      = np.linalg.norm(segment.state.conditions.frames.inertial.velocity_vector,axis=1) 
    alpha      = segment.state.unknowns.wind_angle[:,0][:,None]
    theta      = segment.state.unknowns.body_angle[:,0][:,None]
    
    # Flight path angle
    gamma = theta-alpha

    # process
    v_x =  v_mag * np.cos(gamma)
    v_z = -v_mag * np.sin(gamma) # z points down

    # pack
    conditions.frames.inertial.velocity_vector[:,0] = v_x[:,0]
    conditions.frames.inertial.velocity_vector[:,2] = v_z[:,0]

    return conditions