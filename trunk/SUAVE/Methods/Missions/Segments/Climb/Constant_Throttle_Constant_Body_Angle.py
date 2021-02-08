## @ingroup Methods-Missions-Segments-Climb
# Constant_Throttle_Constant_Speed.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Unpack Unknowns
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Climb
def unpack_unknowns(segment):
    """Unpacks and sets the proper value for body angle

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    state.unknowns.body_angle                      [Radians]

    Outputs:
    state.conditions.frames.body.inertial_rotation [Radians]

    Properties Used:
    N/A
    """          

    # unpack unknowns
    #alpha = segment.state.unknowns.wind_angle
    v_x      = np.vstack([segment.velocity_x_start, segment.state.unknowns.velocity_x])
    v_z      = -np.vstack([segment.velocity_z_start, segment.state.unknowns.velocity_z])
    
    # Flight path angle
    theta = segment.body_angle
    #v_mag = np.linalg.norm(np.hstack([v_x,v_z]),axis=1)[:,None]
    #gamma = np.arccos(v_x/v_mag)
    
    
    #gamma = theta-alpha    
    
    #v_mag = v_x / np.cos(gamma)

    ##v_x =  v_mag * np.cos(gamma)
    #v_z = -v_mag * np.sin(gamma) # z points down    

    # apply unknowns
    segment.state.conditions.frames.body.inertial_rotations[:,1] = theta     

    # pack
    segment.state.conditions.frames.inertial.velocity_vector[:,0] = v_x[:,0]
    segment.state.conditions.frames.inertial.velocity_vector[:,2] = v_z[:,0]


# ----------------------------------------------------------------------
#  Initialize Conditions
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
    throttle   = segment.throttle
    velocity_x_start  = segment.velocity_x_start 
    if segment.velocity_z_start is None:
        velocity_z_start = None
    else:
        velocity_z_start  = -segment.velocity_z_start   
    alt0       = segment.altitude_start 
    altf       = segment.altitude_end
    t_nondim   = segment.state.numerics.dimensionless.control_points
    conditions = segment.state.conditions  
    N        = segment.state.numerics.number_control_points

    # check for initial altitude
    if alt0 is None:
        if not segment.state.initials: raise AttributeError('initial altitude not set')
        alt0 = -1.0 *segment.state.initials.conditions.frames.inertial.position_vector[-1,2]
        
    # check for initial velocity
    # check for initial altitude
    if velocity_x_start is None:
        if not segment.state.initials: raise AttributeError('initial speed not set')
        velocity_x_start = segment.state.initials.conditions.freestream.velocity[-1,0]
        segment.velocity_x_start = velocity_x_start
        velocity_z_start = 0.      
        segment.velocity_z_start = velocity_z_start
        #if alt0 != 0.:
            #raise NotImplementedError
        
    t_guess = segment.state.unknowns.time
        
    initialize_z_vel = altf/t_guess
        
    ones_row_m1 = segment.state.ones_row_m1
    segment.state.unknowns.velocity_x = ones_row_m1(1) * velocity_x_start
    segment.state.unknowns.velocity_z = -ones_row_m1(1) * initialize_z_vel
        
    # Setup the size of the residuals
    #ones_row_m1 = segment.state.ones_row_m1
    #segment.state.residuals.forces = ones_row_m1(1) * 0.0    

    # pack conditions  
    conditions.propulsion.throttle[:,0] = throttle
    conditions.frames.inertial.velocity_vector[:,0] = velocity_x_start # start up value
    conditions.frames.inertial.velocity_vector[:,2] = -initialize_z_vel # start up value

## @ingroup Methods-Missions-Segments-Climb
def update_differentials_altitude(segment):
    """On each iteration creates the differentials and integration funcitons from knowns about the problem. Sets the time at each point. Must return in dimensional time, with t[0] = 0
    
    Assumptions:
    Constant throttle setting, with a constant true airspeed.

    Source:
    N/A

    Inputs:
    segment.climb_angle                         [radians]
    state.conditions.frames.inertial.velocity_vector [meter/second]
    segment.altitude_start                      [meters]
    segment.altitude_end                        [meters]

    Outputs:
    state.conditions.frames.inertial.time       [seconds]
    conditions.frames.inertial.position_vector  [meters]
    conditions.freestream.altitude              [meters]

    Properties Used:
    N/A
    """   

    # unpack
    t = segment.state.numerics.dimensionless.control_points
    D = segment.state.numerics.dimensionless.differentiate
    I = segment.state.numerics.dimensionless.integrate

    
    # Unpack segment initials
    alt0       = segment.altitude_start 
    altf       = segment.altitude_end    
    conditions = segment.state.conditions  
    v          = segment.state.conditions.frames.inertial.velocity_vector
    
    # check for initial altitude
    if alt0 is None:
        if not segment.state.initials: raise AttributeError('initial altitude not set')
        alt0 = -1.0 *segment.state.initials.conditions.frames.inertial.position_vector[-1,2]    
    
    # get overall time step
    vz = -v[:,2,None] # Inertial velocity is z down
    dz = altf- alt0    
    #dt = dz / np.dot(I[-1,:],vz)[-1] # maintain column array
    dt = segment.state.unknowns.time
    
    # Integrate vz to get altitudes
    alt = alt0 + np.dot(I*dt,vz)

    # rescale operators
    t = t * dt

    # pack
    t_initial = segment.state.conditions.frames.inertial.time[0,0]
    segment.state.conditions.frames.inertial.time[:,0] = t_initial + t[:,0]
    conditions.frames.inertial.position_vector[:,2] = -alt[:,0] # z points down
    conditions.freestream.altitude[:,0]             =  alt[:,0] # positive altitude in this context    

    return

# ----------------------------------------------------------------------
#  Update Velocity Vector from Wind Angle
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Climb
def update_velocity_vector_from_wind_angle(segment):
    
    # unpack
    conditions = segment.state.conditions 
    v_x      = np.vstack([segment.velocity_x_start, segment.state.unknowns.velocity_x])
    v_z      = -np.vstack([segment.velocity_z_start, segment.state.unknowns.velocity_z])
    ##v_mag      = np.linalg.norm(segment.state.conditions.frames.inertial.velocity_vector,axis=1) 
    #alpha      = segment.state.unknowns.wind_angle[:,0][:,None]
    #theta      = segment.body_angle
    
    ## Flight path angle
    #gamma = theta-alpha

    ## process
    #v_mag = v_x / np.cos(gamma)
    ##v_x =  v_mag * np.cos(gamma)
    #v_z = -v_mag * np.sin(gamma) # z points down

    # pack
    conditions.frames.inertial.velocity_vector[:,0] = v_x[:,0]
    conditions.frames.inertial.velocity_vector[:,2] = v_z[:,0]

    return conditions

def solve_residuals(segment):
    """ Calculates a residual based on forces
    
        Assumptions:
        The vehicle accelerates, residual on forces and to get it to the final speed
        
        Inputs:
        segment.air_speed_end                  [meters/second]
        segment.state.conditions:
            frames.inertial.total_force_vector [Newtons]
            frames.inertial.velocity_vector    [meters/second]
            weights.total_mass                 [kg]
        segment.state.numerics.time.differentiate
            
        Outputs:
        segment.state.residuals:
            forces               [meters/second^2]
            final_velocity_error [meters/second]
        segment.state.conditions:
            conditions.frames.inertial.acceleration_vector [meters/second^2]

        Properties Used:
        N/A
                                
    """    

    # unpack inputs
    conditions = segment.state.conditions
    FT = conditions.frames.inertial.total_force_vector
    #vi = segment.velocity_start
    v  = conditions.frames.inertial.velocity_vector
    D  = segment.state.numerics.time.differentiate
    m  = conditions.weights.total_mass
    final_alt = -conditions.frames.inertial.position_vector[-1,2]

    # process and pack
    acceleration = np.dot(D , v)
    conditions.frames.inertial.acceleration_vector = acceleration
    
    a  = segment.state.conditions.frames.inertial.acceleration_vector

    segment.state.residuals.force_x = FT[1:,0]/m[1:,0] - a[1:,0]
    segment.state.residuals.force_y = FT[1:,2]/m[1:,0] - a[1:,2]   
    segment.state.residuals.final_altitude = (final_alt - segment.altitude_end)
    
    if np.abs(segment.state.residuals.force_x[0]) < 1e-2:
        aa = 0

    return