## @ingroup Methods-Missions-Segments-Climb
# Constant_Speed_Constant_Angle.py
# 
# Created:  Feb 2018, E. Botero, W. Maier
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np

# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------
## @ingroup Methods-Missions-Segments-Climb
def initialize_conditions(segment,state):
    """Sets the specified conditions which are given for the segment type.
    
    Assumptions:
    Constant Mach number, with a constant angle of climb
    
    Source:
    N/A
    
    Inputs:
    segment.climb_angle                         [Rads]
    segment.air_speed_start                     [m/s]
    segment.altitude_start                      [m]
    segment.throttle_start                      [-]
    segmnet.throttle_end                        [-]
    segment.flight_time                         [-]
    state.numerics.dimensionless.control_points [-]
    
    Outputs:
    conditions.frames.inertial.velocity_vector  [meters/second]
    conditions.frames.inertial.position_vector  [meters]
    conditions.freestream.altitude              [meters]
    
    Properties Used:
    N/A
    """
    
    # Unpack
    flight_time = segment.flight_time
    throttle0   = segment.throttle_start
    throttlef   = segment.throttle_end  
    t_initial   = state.conditions.frames.inertial.time[0,0]
    t_nondim    = state.numerics.dimensionless.control_points
    conditions  = state.conditions
    
    # Setting up time
    time       = t_nondim*flight_time+t_initial     
         
    # Set the throttle
    throttle    = t_nondim*(throttlef-throttle0)+throttle0

    # Pack conditions
    conditions.frames.inertial.time[:,0]      = time[:,0]
    state.conditions.propulsion.throttle[:,0] = throttle[:,0]
    
       
# ----------------------------------------------------------------------
#  Unpack Unknowns
# ----------------------------------------------------------------------
## @ingroup Methods-Missions-Segments-Climb
def unpack_unknowns(segment,state):
    """Unpacks the unknowns set in the mission to be available for the mission.

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    state.unknowns.body_angle          [Rads]
    state.unknowns.velocity            [m/s]
    segment.flight_path_angle          [Rads]

    Outputs:
    conditions.frames.inertial.velocity_vector  [m/s]
    conditions.frames.body.inertial_rotations   [Rads]
    conditions.freestream.altitude              [m]
    conditions.frames.inertial.position_vector  [m]

    
    Properties Used:
    N/A
    """        
    
    # Unpack
    gamma   = segment.flight_path_angle
    speed0  = segment.air_speed_start
    alt0    = segment.altitude_start
    I       = state.numerics.dimensionless.integrate
    
    # Unpack unknowns
    theta   = state.unknowns.body_angle 
    speed   = state.unknowns.air_speed
    
    # check for initial altitude
    if alt0 is None:
        if not state.initials: raise AttributeError('altitude not set')
        alt0 = -1.0 * state.initials.conditions.frames.inertial.position_vector[-1,2]    
    
    # Set initial speed
    speed[0] = speed0
       
    # Find Velocity Vector
    vx     = speed*np.cos(gamma)
    vz     = speed*np.sin(gamma)
    
    # Integrate velocities to get altitudes
    alt = alt0 + np.dot(I,vz)
        
    # Apply unknowns
    state.conditions.frames.body.inertial_rotations[:,1]  = theta[:,0]      
    state.conditions.frames.inertial.position_vector[:,2]       = -alt[:,0] # z points down
    state.conditions.freestream.altitude[:,0]                   = alt[:,0] # positive altitude in this context    
    state.conditions.frames.inertial.velocity_vector[:,0]       = vx[:,0]
    state.conditions.frames.inertial.velocity_vector[:,2]       = vz[:,0]
    