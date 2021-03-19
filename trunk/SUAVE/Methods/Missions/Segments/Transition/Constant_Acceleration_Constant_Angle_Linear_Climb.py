## @ingroup Methods-Missions-Segments-Transition
# Constant_Acceleration_Constant_Angle_Linear_Climb.py
# 
# Created:  Feb 2019, M. Clarke

# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------
import numpy as np

## @ingroup Methods-Missions-Segments-Transition
def initialize_conditions(segment):
    """Sets the specified conditions which are given for the segment type.

    Assumptions:
    Constant acceleration and constant altitude

    Source:
    N/A

    Inputs:
    segment.altitude                [meters]
    segment.air_speed_start         [meters/second]
    segment.air_speed_end           [meters/second]
    segment.acceleration            [meters/second^2]
    conditions.frames.inertial.time [seconds]

    Outputs:
    conditions.frames.inertial.velocity_vector  [meters/second]
    conditions.frames.inertial.position_vector  [meters]
    conditions.freestream.altitude              [meters]
    conditions.frames.inertial.time             [seconds]

    Properties Used:
    N/A
    """      
    
    # unpack
    alt0        = segment.altitude_start 
    altf        = segment.altitude_end 
    climb_angle = segment.climb_angle
    v0          = segment.air_speed 
    ax          = segment.acceleration   
    T0          = segment.pitch_initial
    Tf          = segment.pitch_final     
    t_nondim    = segment.state.numerics.dimensionless.control_points
    conditions  = segment.state.conditions 
    
    # check for climb angle     
    if climb_angle is None:
        raise AttributeError('set climb')
    
    if ax is None: 
        raise AttributeError('set acceleration') 
    
    # check for initial and final altitude 
    if alt0 is None:
        if not segment.state.initials: raise AttributeError('altitude not set')
        alt0 = -1.0 * segment.state.initials.conditions.frames.inertial.position_vector[-1,2] 
    
    if altf is None:
        raise AttributeError('final altitude not set')
        
    # check for initial pitch
    if T0 is None:
        T0  =  segment.state.initials.conditions.frames.body.inertial_rotations[-1,1] 
        
    # check for initial velocity vector
    if v0 is None:
        v0  =  segment.state.initials.conditions.frames.inertial.velocity_vector[-1,:] 
        segment.velocity_vector = v0
        
         
    # discretize on altitude
    v0_mag          = np.linalg.norm(v0)
    alt             = t_nondim * (altf-alt0) + alt0   
    ground_distance = abs(altf-alt0)/np.tan(climb_angle)
    true_distance   = np.sqrt((altf-alt0)**2 + ground_distance**2)
    t_initial       = conditions.frames.inertial.time[0,0]   
    elapsed_time    = (-v0_mag + np.sqrt(v0_mag**2 + 2*ax*true_distance))/(ax) 
    vf_mag          = v0_mag + ax*(elapsed_time)   
    
    # dimensionalize time        
    t_final   = t_initial + elapsed_time
    t_nondim  = segment.state.numerics.dimensionless.control_points
    time      = t_nondim * (t_final-t_initial)
    
    # Figure out vx
    V  = (vf_mag-v0_mag) 
    vx = t_nondim *  V  * np.cos(climb_angle) + v0 * np.cos(climb_angle) 
    vz = t_nondim *  V  * np.sin(climb_angle) + v0 * np.sin(climb_angle)  
    
    # set the body angle
    body_angle =time*(Tf-T0)/(t_final-t_initial) + T0
    segment.state.conditions.frames.body.inertial_rotations[:,1] = body_angle[:,0]     
    
    # pack
    segment.state.conditions.freestream.altitude[:,0] = alt[:,0]
    segment.state.conditions.frames.inertial.position_vector[:,2] = -alt[:,0] # z points down
    segment.state.conditions.frames.inertial.velocity_vector[:,0] = vx[:,0] 
    segment.state.conditions.frames.inertial.velocity_vector[:,2] = -vz[:,0] 
    segment.state.conditions.frames.inertial.time[:,0] = time[:,0]
        
# ----------------------------------------------------------------------
#  Residual Total Forces
# ----------------------------------------------------------------------
    
## @ingroup Methods-Missions-Segments-Cruise    
def residual_total_forces(segment):
    """ Calculates a residual based on forces
    
        Assumptions:
        The vehicle is not accelerating, doesn't use gravity
        
        Inputs:
            segment.acceleration                   [meters/second^2]
            segment.state.ones_row                 [vector]
            state.conditions:
                frames.inertial.total_force_vector [Newtons]
                weights.total_mass                 [kg]
            
        Outputs:
            state.conditions:
                state.residuals.forces [meters/second^2]

        Properties Used:
        N/A
                                
    """      
    
    # Unpack
    FT      = segment.state.conditions.frames.inertial.total_force_vector
    v       = segment.state.conditions.frames.inertial.velocity_vector
    D       = segment.state.numerics.time.differentiate      
    a       = np.dot(D,v)
    segment.state.conditions.frames.inertial.acceleration_vector = a      
    m       = segment.state.conditions.weights.total_mass

       
    # horizontal
    segment.state.residuals.forces[:,0] = FT[:,0]/m[:,0] - a[:,0]
    # vertical
    segment.state.residuals.forces[:,1] = FT[:,2]/m[:,0] - a[:,2]
     
    return
