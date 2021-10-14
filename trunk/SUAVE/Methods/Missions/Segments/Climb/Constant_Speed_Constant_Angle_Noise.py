## @ingroup Methods-Missions-Segments-Climb
# Constant_Speed_Constant_Angle_Noise.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np

# ----------------------------------------------------------------------
#  Expand State
# ----------------------------------------------------------------------
## @ingroup Methods-Missions-Segments-Climb
def expand_state(segment):
    
    """Makes all vectors in the state the same size. Determines the minimum amount of points needed to get data for noise certification.

    Assumptions:
    Half second intervals for certification requirements. Fixed microphone position

    Source:
    N/A

    Inputs:
    state.numerics.number_control_points  [Unitless]

    Outputs:
    N/A

    Properties Used:
    Position of the flyover microphone is 6500 meters
    """          
    
    # unpack
    climb_angle  = segment.climb_angle
    air_speed    = segment.air_speed   
    conditions   = segment.state.conditions
    
    #Necessary input for determination of noise trajectory    
    dt = 0.5  #time step in seconds for noise calculation - Certification requirement    
    x0 = 6500 #Position of the Flyover microphone relatively to the break-release point
    
    # process velocity vector
    v_x=air_speed*np.cos(climb_angle)
    
    #number of time steps (space discretization)
    total_time=(x0+500)/v_x    
    n_points   = np.int(np.ceil(total_time/dt +1))       
    
    segment.state.numerics.number_control_points = n_points
    
    segment.state.expand_rows(n_points,override=True)      
    
    return

# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------
## @ingroup methods-missions-segments-climb
def initialize_conditions(segment):
    """Gets the overall time step for the segment type.
    
    Assumptions:
    Constant true airspeed, with a constant climb angle. This segment is specically created for noise calculations.

    Source:
    N/A

    Inputs:
    segment.climb_angle                         [radians]
    segment.air_speed                           [meter/second]
    segment.altitude_start                      [meters]
    segment.altitude_end                        [meters]
    state.numerics.dimensionless.control_points [Unitless]
    conditions.freestream.density               [kilograms/meter^3]

    Outputs:
    conditions.frames.inertial.velocity_vector  [meters/second]
    conditions.frames.inertial.position_vector  [meters]
    conditions.freestream.altitude              [meters]

    Properties Used:
    N/A
    """     
    
    dt=0.5  #time step in seconds for noise calculation
    
    # unpack
    climb_angle = segment.climb_angle
    air_speed   = segment.air_speed   
    t_nondim    = segment.state.numerics.dimensionless.control_points
    conditions  = segment.state.conditions  
    
    # process velocity vector
    v_mag = air_speed
    v_x   = v_mag * np.cos(climb_angle)
    v_z   = -v_mag * np.sin(climb_angle)    

    #initial altitude
    alt0 = 10.668   #(35ft)
    altf = alt0 + (-v_z)*dt*len(t_nondim)

    # discretize on altitude
    alt = t_nondim * (altf-alt0) + alt0    
    
    # pack conditions    
    conditions.frames.inertial.velocity_vector[:,0] = v_x
    conditions.frames.inertial.velocity_vector[:,2] = v_z
    conditions.frames.inertial.position_vector[:,2] = -alt[:,0] # z points down
    conditions.freestream.altitude[:,0]             =  alt[:,0] # positive altitude in this context
