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
    climb_angle = segment.climb_angle
    air_speed   = segment.air_speed
    alt0        = segment.altitude_start
    flight_time = segment.flight_time
    throttle0   = segment.throttle_start
    throttlef   = segment.throttle_end
    cntrl_pts   = state.numerics.dimensionless.control_points
    conditions  = state.conditions
    
    # Discretize on time
    time        = cntrl_pts*flight_time
     
    # Set the throttle
    throttle    = cntrl_pts*(throttlef-throttle0)+throttle0
    
    # Process velocity vector
    v_mag = air_speed
    v_x   = v_mag * np.cos(climb_angle)
    v_z   = -v_mag * np.sin(climb_angle)
    
    # Pack conditions
    conditions.frames.inertial.velocity_vector[:,0] = v_x
    conditions.frames.inertial.velocity_vector[:,2] = v_z
    conditions.frames.inertial.position_vector[:,2] = -alt[:,0] # z points down
    conditions.freestream.altitude[:,0]             =  alt[:,0] # positive altitude in this context
    conditions.freestream.time[:,0]                 =  time[:,0]
    state.conditions.propulsion.throttle[:,0]       = throttle[:,0]
    
       
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


    Outputs:
    state.conditions.propulsion.throttle            [-]
    state.conditions.frames.body.inertial_rotations [Rads]

    Properties Used:
    N/A
    """        
    
    # Unpack unknowns
    theta    = self.state.unknowns.body_angle 
    velocity = self.state.unknowns.velocity
       

    # integrate velocities to get altitudes
    
    #
    
    # apply unknowns
    state.conditions.frames.body.inertial_rotations[:,1]  = theta[:,0]      
    state.conditions.frames.inertial.velocity_vector[:,:] = velocity
    