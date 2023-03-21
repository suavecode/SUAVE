# Linear_Mach_Constant_Rate.py
# 
# Created:  Jul 2014, SUAVE Team (Stanford University)
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
import MARC

# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------

def initialize_conditions(segment):
    """Sets the specified conditions which are given for the segment type.

    Assumptions:
    Change mach linearly through the descent with constant descent rate

    Source:
    N/A

    Inputs:
    segment.descent_rate                        [meters/second]
    segment.altitude_start                      [meters]
    segment.altitude_end                        [meters]
    segment.mach_number_start                          [unitless]
    segment.mach_number_end                            [unitless]
    segment.air_speed                           [meters/second]
    state.numerics.dimensionless.control_points [array]

    Outputs:
    conditions.frames.inertial.velocity_vector  [meters/second]
    conditions.frames.inertial.position_vector  [meters]
    conditions.freestream.altitude              [meters]
    conditions.frames.inertial.time             [seconds]

    Properties Used:
    N/A
    """      
    
    # unpack
    descent_rate = segment.descent_rate
    M0           = segment.mach_number_start
    Mf           = segment.mach_number_end
    alt0         = segment.altitude_start 
    altf         = segment.altitude_end
    t_nondim     = segment.state.numerics.dimensionless.control_points
    conditions   = segment.state.conditions  

    # Update freestream to get speed of sound
    MARC.Methods.Missions.Segments.Common.Aerodynamics.update_atmosphere(segment)
    a          = conditions.freestream.speed_of_sound        
    
    # check for initial altitude
    if alt0 is None:
        if not segment.state.initials: raise AttributeError('initial altitude not set')
        alt0 = -1.0 *segment.state.initials.conditions.frames.inertial.position_vector[-1,2]
    
    # check for initial velocity vector
    if M0 is None:
        if not segment.state.initials: raise AttributeError('initial mach number not set')
        M0  =  np.linalg.norm(segment.state.initials.conditions.frames.inertial.velocity_vector[-1,:])/a[0,:]         
        
    # discretize on altitude
    alt = t_nondim * (altf-alt0) + alt0
    conditions.freestream.altitude[:,0] =  alt[:,0]  # positive altitude 

    # process velocity vector
    mach_number = (Mf-M0)*t_nondim + M0
    v_mag       = mach_number * a
    v_z         = descent_rate # z points down
    v_x         = np.sqrt( v_mag**2 - v_z**2 )
    
    # pack conditions    
    conditions.frames.inertial.velocity_vector[:,0] = v_x[:,0]
    conditions.frames.inertial.velocity_vector[:,2] = v_z
    conditions.frames.inertial.position_vector[:,2] = -alt[:,0] # z points down
    conditions.freestream.altitude[:,0]             =  alt[:,0] # positive altitude t
