## @ingroup Methods-Missions-Segments-Climb
# Constant_Mach_Constant_Angle.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero
#           Jun 2017, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
import SUAVE
# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------
## @ingroup Methods-Missions-Segments-Climb
def initialize_conditions(segment):
    """Sets the specified conditions which are given for the segment type.
    
    Assumptions:
    Constant Mach number, with a constant angle of climb

    Source:
    N/A

    Inputs:
    segment.climb_angle                                 [radians]
    segment.mach                                        [Unitless]
    segment.altitude_start                              [meters]
    segment.altitude_end                                [meters]
    segment.state.numerics.dimensionless.control_points [Unitless]
    conditions.freestream.density                       [kilograms/meter^3]

    Outputs:
    conditions.frames.inertial.velocity_vector  [meters/second]
    conditions.frames.inertial.position_vector  [meters]
    conditions.freestream.altitude              [meters]

    Properties Used:
    N/A
    """       
    # unpack user inputs
    climb_angle = segment.climb_angle
    mach_number = segment.mach
    alt0        = segment.altitude_start 
    altf        = segment.altitude_end
    t_nondim    = segment.state.numerics.dimensionless.control_points
    conditions  = segment.state.conditions 
    
    # unpack unknowns
    throttle = segment.state.unknowns.throttle
    theta    = segment.state.unknowns.body_angle
    alts     = segment.state.unknowns.altitudes   
    
    # check for initial altitude
    if alt0 is None:
        if not segment.state.initials: raise AttributeError('initial altitude not set')
        alt0 = -1.0 * segment.state.initials.conditions.frames.inertial.position_vector[-1,2]
    
    # pack conditions   
    conditions.freestream.altitude[:,0]             =  alts[:,0] # positive altitude in this context
    
    # Update freestream to get speed of sound
    SUAVE.Methods.Missions.Segments.Common.Aerodynamics.update_atmosphere(segment)
    a = conditions.freestream.speed_of_sound    
    
    # process velocity vector
    v_mag = mach_number * a
    v_x   = v_mag * np.cos(climb_angle)
    v_z   = -v_mag * np.sin(climb_angle)
    
    # pack conditions    
    conditions.frames.inertial.velocity_vector[:,0] = v_x[:,0]
    conditions.frames.inertial.velocity_vector[:,2] = v_z[:,0]
    conditions.frames.inertial.position_vector[:,2] = -alts[:,0] # z points down
    segment.state.conditions.propulsion.throttle[:,0]            = throttle[:,0]
    segment.state.conditions.frames.body.inertial_rotations[:,1] = theta[:,0]  
    
def residual_total_forces(segment):
    
    # Unpack results
    FT = segment.state.conditions.frames.inertial.total_force_vector
    a  = segment.state.conditions.frames.inertial.acceleration_vector
    m  = segment.state.conditions.weights.total_mass    
    alt_in  = segment.state.unknowns.altitudes[:,0] 
    alt_out = segment.state.conditions.freestream.altitude[:,0] 
    
    # Residual in X and Z, as well as a residual on the guess altitude
    segment.state.residuals.forces[:,0] = FT[:,0]/m[:,0] - a[:,0]
    segment.state.residuals.forces[:,1] = FT[:,2]/m[:,0] - a[:,2]
    segment.state.residuals.forces[:,2] = (alt_in - alt_out)/alt_out[-1]

    return    