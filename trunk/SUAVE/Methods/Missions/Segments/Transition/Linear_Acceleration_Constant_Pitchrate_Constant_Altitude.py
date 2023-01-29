## @ingroup Methods-Missions-Segments-Transition
# Linear_Acceleration_Constant_Pitchrate_Constant_Altitude.py
# 
# Created:  Jan 2023, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------
import numpy as np 
## @ingroup Methods-Missions-Segments-Transition
def initialize_conditions(segment):
    """Sets the specified conditions which are given for the segment type.

    Assumptions:
    Linear acceleration, constant pitch rate, and constant altitude

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
    alt = segment.altitude 
    v0  = segment.air_speed_start
    vf  = segment.air_speed_end  
    ax0 = segment.acceleration_initial 
    axf = segment.acceleration_final 
    T0  = segment.pitch_initial
    Tf  = segment.pitch_final     
    
    # check for initial altitude
    if alt is None:
        if not segment.state.initials: raise AttributeError('altitude not set')
        alt = -1.0 * segment.state.initials.conditions.frames.inertial.position_vector[-1,2]
        segment.altitude = alt
        
    # check for initial pitch
    if T0 is None:
        T0  =  segment.state.initials.conditions.frames.body.inertial_rotations[-1,1]
        segment.pitch_initial = T0    
    
    # compute control point accelerations
    t_nondim = segment.state.numerics.dimensionless.control_points  
    n_cp     = len(t_nondim)
    ax_t     = ax0 + (t_nondim * (axf-ax0))  # linear acceleration
    
    # set up A matrix for solving control point delta_t and velocities
    Amat = np.identity(n_cp-1)
    i,j  = np.indices(Amat.shape)
    Amat[i==j+1] = -1 * np.ones(n_cp-2)
    Amat[j==n_cp-2] = -0.5 * np.ravel((ax_t[1:]-ax_t[0:-1])) - np.ravel(ax_t[0:-1])
    
    # setup the b solution vector
    b = np.zeros(n_cp-1)
    b[0] = v0
    b[-1] = -vf
    
    # solve for x
    x = np.linalg.solve(Amat, b)
    vx_t = np.atleast_2d(np.hstack((v0, x[0:-1], vf))).T
    dt = x[-1]

    # dimensionalize time
    t_initial = segment.state.conditions.frames.inertial.time[0,0]    
    t_final   = t_initial + (n_cp - 1) * dt
    time      = t_nondim * (t_final-t_initial) + t_initial
    
    # Figure out x
    x0 = segment.state.conditions.frames.inertial.position_vector[:,0]
    xpos = x0 + (vx_t[:,0] * time[:,0])
    
    # set the body angle
    body_angle = T0 + time*(Tf-T0)/(t_final-t_initial) 
    segment.state.conditions.frames.body.inertial_rotations[:,1] = body_angle[:,0]     
    
    # pack
    segment.state.conditions.freestream.altitude[:,0] = alt
    segment.state.conditions.frames.inertial.position_vector[:,2] = -alt # z points down
    segment.state.conditions.frames.inertial.position_vector[:,0] = xpos # z points down    
    segment.state.conditions.frames.inertial.velocity_vector[:,0] = vx_t[:,0]
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
    FT       = segment.state.conditions.frames.inertial.total_force_vector
    ax0      = segment.acceleration_initial
    axf      = segment.acceleration_final
    m        = segment.state.conditions.weights.total_mass  
    t_nondim = segment.state.numerics.dimensionless.control_points  
    
    a_x     = ax0 + (t_nondim * (axf-ax0))  # linear acceleration
    
    # horizontal
    segment.state.residuals.forces[:,0] = FT[:,0]/m[:,0] - a_x[:,0]
    # vertical
    segment.state.residuals.forces[:,1] = FT[:,2]/m[:,0] 

    return
