## @ingroup Methods-Missions-Segments-Climb
# Constant_Mach_Linear_Altitude.py

# Created:  Jul 2014, SUAVE Team (Stanford University)
# Modified: Jun 2017, E. Botero 

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
import MARC
import numpy as np

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
    segment.mach_number                         [unitless]
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
    mach       = segment.mach_number
    conditions = segment.state.conditions  
    t_nondim   = segment.state.numerics.dimensionless.control_points
    
    # check for initial altitude
    if alt0 is None:
        if not segment.state.initials: raise AttributeError('altitude not set')
        alt0 = -1.0 * segment.state.initials.conditions.frames.inertial.position_vector[-1,2]
        
    # discretize on altitude
    alt = t_nondim * (altf-alt0) + alt0      
    segment.state.conditions.freestream.altitude[:,0] = alt[:,0]

    # check for initial velocity
    if mach is None: 
        if not segment.state.initials: raise AttributeError('mach not set')
        air_speed = np.linalg.norm(segment.state.initials.conditions.frames.inertial.velocity_vector[-1])   
    else:
        
        # Update freestream to get speed of sound
        MARC.Methods.Missions.Segments.Common.Aerodynamics.update_atmosphere(segment)
        a          = conditions.freestream.speed_of_sound    
    
        # compute speed, constant with constant altitude
        air_speed    = mach * a   
        
    climb_angle  = np.arctan((altf-alt0)/xf)
    v_x          = np.cos(climb_angle)*air_speed
    v_z          = np.sin(climb_angle)*air_speed 
    t_nondim     = segment.state.numerics.dimensionless.control_points
    
    # discretize on altitude
    alt = t_nondim * (altf-alt0) + alt0    
    
    # pack
    conditions.freestream.altitude[:,0]             = alt[:,0]
    conditions.frames.inertial.position_vector[:,2] = -alt[:,0] # z points down 
    conditions.frames.inertial.velocity_vector[:,0] = v_x[:,0]
    conditions.frames.inertial.velocity_vector[:,2] = -v_z[:,0]      