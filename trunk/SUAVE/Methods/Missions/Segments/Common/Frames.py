## @ingroup Methods-Missions-Segments-Common
# Frames.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jul 2016, E. Botero
#           Jul 2017, E. Botero
#           May 2019, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Core import Units

from SUAVE.Methods.Geometry.Three_Dimensional \
     import angles_to_dcms, orientation_product, orientation_transpose

# ----------------------------------------------------------------------
#  Initialize Inertial Position
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Common
def initialize_inertial_position(segment):
    """ Sets the initial location of the vehicle at the start of the segment
    
        Assumptions:
        Only used if there is an initial condition
        
        Inputs:
            segment.state.initials.conditions:
                frames.inertial.position_vector   [meters]
            segment.state.conditions:           
                frames.inertial.position_vector   [meters]
            
        Outputs:
            segment.state.conditions:           
                frames.inertial.position_vector   [meters]

        Properties Used:
        N/A
                                
    """    
    
    if segment.state.initials:
        r_initial = segment.state.initials.conditions.frames.inertial.position_vector
        r_current = segment.state.conditions.frames.inertial.position_vector
        
        if 'altitude' in segment.keys() and segment.altitude:
            r_initial[-1,None,-1] = -segment.altitude
        elif 'altitude_start' in segment.keys() and segment.altitude_start:
            r_initial[-1,None,-1] = -segment.altitude_start
            
        segment.state.conditions.frames.inertial.position_vector[:,:] = r_current + (r_initial[-1,None,:] - r_current[0,None,:])
        
    return
    
    
# ----------------------------------------------------------------------
#  Initialize Time
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Common
def initialize_time(segment):
    """ Sets the initial time of the vehicle at the start of the segment
    
        Assumptions:
        Only used if there is an initial condition
        
        Inputs:
            state.initials.conditions:
                frames.inertial.time     [seconds]
                frames.planet.start_time [seconds]
            state.conditions:           
                frames.inertial.time     [seconds]
            segment.start_time           [seconds]
            
        Outputs:
            state.conditions:           
                frames.inertial.time     [seconds]
                frames.planet.start_time [seconds]

        Properties Used:
        N/A
                                
    """        
    
    if segment.state.initials:
        t_initial = segment.state.initials.conditions.frames.inertial.time
        t_current = segment.state.conditions.frames.inertial.time
        
        segment.state.conditions.frames.inertial.time[:,:] = t_current + (t_initial[-1,0] - t_current[0,0])
        
    else:
        t_initial = segment.state.conditions.frames.inertial.time[0,0]
        
    if segment.state.initials:
        segment.state.conditions.frames.planet.start_time = segment.state.initials.conditions.frames.planet.start_time
        
    elif 'start_time' in segment:
        segment.state.conditions.frames.planet.start_time = segment.start_time
    
    return
    

# ----------------------------------------------------------------------
#  Initialize Planet Position
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Common
def initialize_planet_position(segment):
    """ Sets the initial location of the vehicle relative to the planet at the start of the segment
    
        Assumptions:
        Only used if there is an initial condition
        
        Inputs:
            segment.state.initials.conditions:
                frames.planet.longitude [Radians]
                frames.planet.latitude  [Radians]
            segment.longitude           [Radians]
            segment.latitude            [Radians]

        Outputs:
            segment.state.conditions:           
                frames.planet.latitude  [Radians]
                frames.planet.longitude [Radians]

        Properties Used:
        N/A
                                
    """        
    
    if segment.state.initials:
        longitude_initial = segment.state.initials.conditions.frames.planet.longitude[-1,0]
        latitude_initial  = segment.state.initials.conditions.frames.planet.latitude[-1,0] 
    elif 'latitude' in segment:
        longitude_initial = segment.longitude
        latitude_initial  = segment.latitude      
    else:
        longitude_initial = 0.0
        latitude_initial  = 0.0


    segment.state.conditions.frames.planet.longitude[:,0] = longitude_initial
    segment.state.conditions.frames.planet.latitude[:,0]  = latitude_initial    

    return
    
    
# ----------------------------------------------------------------------
#  Update Planet Position
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Common
def update_planet_position(segment):
    """ Updates the location of the vehicle relative to the Planet throughout the mission
    
        Assumptions:
        This is valid for small movements and times as it does not account for the rotation of the Planet beneath the vehicle
        
        Inputs:
        segment.state.conditions:
            freestream.velocity                      [meters/second]
            freestream.altitude                      [meters]
            frames.body.inertial_rotations           [Radians]
        segment.analyses.planet.features.mean_radius [meters]
        segment.state.numerics.time.integrate        [float]
            
        Outputs:
            segment.state.conditions:           
                frames.planet.latitude  [Radians]
                frames.planet.longitude [Radians]

        Properties Used:
        N/A
                                
    """        
    
    # unpack
    conditions = segment.state.conditions
    
    # unpack orientations and velocities
    V          = conditions.freestream.velocity[:,0]
    altitude   = conditions.freestream.altitude[:,0]
    phi        = conditions.frames.body.inertial_rotations[:,0]
    theta      = conditions.frames.body.inertial_rotations[:,1]
    psi        = conditions.frames.body.inertial_rotations[:,2]
    alpha      = conditions.aerodynamics.angle_of_attack[:,0]
    I          = segment.state.numerics.time.integrate
    Re         = segment.analyses.planet.features.mean_radius

    # The flight path and radius
    gamma     = theta - alpha
    R         = altitude + Re

    # Find the velocities and integrate the positions
    lamdadot  = (V/R)*np.cos(gamma)*np.cos(psi)
    lamda     = np.dot(I,lamdadot) / Units.deg # Latitude
    mudot     = (V/R)*np.cos(gamma)*np.sin(psi)/np.cos(lamda)
    mu        = np.dot(I,mudot) / Units.deg # Longitude

    # Reshape the size of the vectorss
    shape     = np.shape(conditions.freestream.velocity)
    mu        = np.reshape(mu,shape)
    lamda     = np.reshape(lamda,shape)

    # Pack'r up
    lat = conditions.frames.planet.latitude[0,0]
    lon = conditions.frames.planet.longitude[0,0]
    conditions.frames.planet.latitude  = lat + lamda
    conditions.frames.planet.longitude = lon + mu

    return
    
    
# ----------------------------------------------------------------------
#  Update Orientations
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Common
def update_orientations(segment):
    
    """ Updates the orientation of the vehicle throughout the mission for each relevant axis
    
        Assumptions:
        This assumes the vehicle has 3 frames: inertial, body, and wind. There also contains bits for stability axis which are not used. Creates tensors and solves for alpha and beta.
        
        Inputs:
        segment.state.conditions:
            frames.inertial.velocity_vector          [meters/second]
            frames.body.inertial_rotations           [Radians]
        segment.analyses.planet.features.mean_radius [meters]
        state.numerics.time.integrate                [float]
            
        Outputs:
            segment.state.conditions:           
                aerodynamics.angle_of_attack      [Radians]
                aerodynamics.side_slip_angle      [Radians]
                aerodynamics.roll_angle           [Radians]
                frames.body.transform_to_inertial [Radians]
                frames.wind.body_rotations        [Radians]
                frames.wind.transform_to_inertial [Radians]
    

        Properties Used:
        N/A
    """

    # unpack
    conditions = segment.state.conditions
    V_inertial = conditions.frames.inertial.velocity_vector
    body_inertial_rotations = conditions.frames.body.inertial_rotations

    # ------------------------------------------------------------------
    #  Body Frame
    # ------------------------------------------------------------------

    # body frame rotations
    phi   = body_inertial_rotations[:,0,None]
    theta = body_inertial_rotations[:,1,None]
    psi   = body_inertial_rotations[:,2,None]

    # body frame tranformation matrices
    T_inertial2body = angles_to_dcms(body_inertial_rotations,(2,1,0))
    T_body2inertial = orientation_transpose(T_inertial2body)

    # transform inertial velocity to body frame
    V_body = orientation_product(T_inertial2body,V_inertial)

    # project inertial velocity into body x-z plane
    V_stability = V_body * 1.
    V_stability[:,1] = 0
    V_stability_magnitude = np.sqrt( np.sum(V_stability**2,axis=1) )[:,None]

    # calculate angle of attack
    alpha = np.arctan2(V_stability[:,2],V_stability[:,0])[:,None]

    # calculate side slip
    beta = np.arctan2(V_body[:,1],V_stability_magnitude[:,0])[:,None]

    # pack aerodynamics angles
    conditions.aerodynamics.angle_of_attack[:,0] = alpha[:,0]
    conditions.aerodynamics.side_slip_angle[:,0] = -beta[:,0]
    conditions.aerodynamics.roll_angle[:,0]      = phi[:,0]

    # pack transformation tensor
    conditions.frames.body.transform_to_inertial = T_body2inertial


    # ------------------------------------------------------------------
    #  Wind Frame
    # ------------------------------------------------------------------

    # back calculate wind frame rotations
    wind_body_rotations = body_inertial_rotations * 0.
    wind_body_rotations[:,0] = 0          # no roll in wind frame
    wind_body_rotations[:,1] = alpha[:,0] # theta is angle of attack
    wind_body_rotations[:,2] = beta[:,0]  # psi is side slip angle

    # wind frame tranformation matricies
    T_wind2body = angles_to_dcms(wind_body_rotations,(2,1,0))
    T_body2wind = orientation_transpose(T_wind2body)
    T_wind2inertial = orientation_product(T_wind2body,T_body2inertial)

    # pack wind rotations
    conditions.frames.wind.body_rotations = wind_body_rotations

    # pack transformation tensor
    conditions.frames.wind.transform_to_inertial = T_wind2inertial
    
    return
        

# ----------------------------------------------------------------------
#  Update Forces
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Common
def update_forces(segment):
    
    """ Summation of forces: lift, drag, thrust, weight. Future versions will include new definitions of dreams, FAA, money, and reality.
    
        Assumptions:
        You only have these 4 forces applied to the vehicle
        
        Inputs:
            segment.state.conditions:
                frames.wind.lift_force_vector          [newtons]
                frames.wind.drag_force_vector          [newtons]
                frames.inertial.gravity_force_vector   [newtons]
                frames.body.thrust_force_vector        [newtons]
                frames.body.transform_to_inertial      [newtons]
                frames.wind.transform_to_inertial      [newtons]

            
        Outputs:
            segment.state.conditions:           
                frames.inertial.total_force_vector [newtons]

    

        Properties Used:
        N/A
    """    

    # unpack
    conditions = segment.state.conditions

    # unpack forces
    wind_lift_force_vector        = conditions.frames.wind.lift_force_vector
    wind_drag_force_vector        = conditions.frames.wind.drag_force_vector
    body_thrust_force_vector      = conditions.frames.body.thrust_force_vector
    inertial_gravity_force_vector = conditions.frames.inertial.gravity_force_vector

    # unpack transformation matrices
    T_body2inertial = conditions.frames.body.transform_to_inertial
    T_wind2inertial = conditions.frames.wind.transform_to_inertial

    # to inertial frame
    L = orientation_product(T_wind2inertial,wind_lift_force_vector)
    D = orientation_product(T_wind2inertial,wind_drag_force_vector)
    T = orientation_product(T_body2inertial,body_thrust_force_vector)
    W = inertial_gravity_force_vector

    # sum of the forces
    F = L + D + T + W
    # like a boss

    # pack
    conditions.frames.inertial.total_force_vector[:,:] = F[:,:]

    return

# ----------------------------------------------------------------------
#  Integrate Position
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Common
def integrate_inertial_horizontal_position(segment):
    """ Determines how far the airplane has traveled. 
    
        Assumptions:
        Assumes a flat earth, this is planar motion.
        
        Inputs:
            segment.state.conditions:
                frames.inertial.position_vector [meters]
                frames.inertial.velocity_vector [meters/second]
            segment.state.numerics.time.integrate       [float]
            
        Outputs:
            segment.state.conditions:           
                frames.inertial.position_vector [meters]

        Properties Used:
        N/A
                                
    """        

    # unpack
    conditions = segment.state.conditions
    x0 = conditions.frames.inertial.position_vector[0,None,0:1+1]
    vx = conditions.frames.inertial.velocity_vector[:,0:1+1]
    I  = segment.state.numerics.time.integrate
    
    # integrate
    x = np.dot(I,vx) + x0
    
    # pack
    conditions.frames.inertial.position_vector[:,0:1+1] = x[:,:]
    
    return

# ----------------------------------------------------------------------
#  Update Acceleration
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Common
def update_acceleration(segment):
    """ Differentiates the velocity vector to get accelerations
    
        Assumptions:
        Assumes a flat earth, this is planar motion.
        
        Inputs:
            segment.state.conditions:
                frames.inertial.velocity_vector     [meters/second]
            segment.state.numerics.time.differentiate       [float]
            
        Outputs:
            segment.state.conditions:           
                frames.inertial.acceleration_vector [meters]

        Properties Used:
        N/A
                                
    """            
    
    # unpack conditions
    v = segment.state.conditions.frames.inertial.velocity_vector
    D = segment.state.numerics.time.differentiate
    
    # accelerations
    acc = np.dot(D,v)
    
    # pack conditions
    segment.state.conditions.frames.inertial.acceleration_vector[:,:] = acc[:,:]   