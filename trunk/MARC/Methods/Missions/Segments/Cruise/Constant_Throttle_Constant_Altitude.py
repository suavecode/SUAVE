## @ingroup Methods-Missions-Segments-Cruise
# Constant_Throttle_Constant_Altitude.py
# 
# Created:  Jul 2014, SUAVE Team (Stanford University)
# Modified: Jan 2016, E. Botero
#           May 2019, T. MacDonald
#           Mar 2020, M. Clarke
#           Apr 2022, A. Blaufox

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Unpack Unknowns
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Cruise
def unpack_unknowns(segment):
    
    # unpack unknowns
    unknowns   = segment.state.unknowns
    theta      = unknowns.body_angle
    accel_x    = unknowns.accel_x
    time       = unknowns.time
    
    # rescale time
    t_initial  = segment.state.conditions.frames.inertial.time[0,0]
    t_final    = t_initial + time  
    t_nondim   = segment.state.numerics.dimensionless.control_points
    time       = t_nondim * (t_final-t_initial) + t_initial     

    # build acceleration
    N          = segment.state.numerics.number_control_points
    a          = np.zeros((N, 3))
    a[:, 0]    = accel_x[:,0]
    
    # apply unknowns
    conditions = segment.state.conditions
    conditions.frames.body.inertial_rotations[:,1]  = theta[:,0]  
    conditions.frames.inertial.acceleration_vector  = a
    conditions.frames.inertial.time[:,0]            = time[:,0]
    
    return 

# ----------------------------------------------------------------------
#  Integrate Velocity
# ---------------------------------------------------------------------- 

def integrate_velocity(segment):
    
    # unpack 
    conditions = segment.state.conditions
    v0         = segment.air_speed_start
    I          = segment.state.numerics.time.integrate
    a          = conditions.frames.inertial.acceleration_vector
    
    # compute x-velocity
    velocity_x = v0 + np.dot(I, a)[:,0]   

    # pack velocity
    conditions.frames.inertial.velocity_vector[:,0] = velocity_x
    
    return

# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------    

## @ingroup Methods-Missions-Segments-Cruise
def initialize_conditions(segment):
    """Sets the specified conditions which are given for the segment type.

    Assumptions:
    Constant throttle and constant altitude, allows for acceleration

    Source:
    N/A

    Inputs:
    segment.altitude                             [meters]
    segment.air_speed_start                      [meters/second]
    segment.air_speed_end                        [meters/second]
    segment.throttle	                         [unitless]
    segment.state.numerics.number_control_points [int]

    Outputs:
    state.conditions.propulsion.throttle        [unitless]
    conditions.frames.inertial.position_vector  [meters]
    conditions.freestream.altitude              [meters]

    Properties Used:
    N/A
    """   
    
    state      = segment.state
    conditions = state.conditions

    # unpack inputs
    alt      = segment.altitude 
    v0       = segment.air_speed_start
    vf       = segment.air_speed_end 
    throttle = segment.throttle	
    N        = segment.state.numerics.number_control_points   
    
    # check for initial altitude
    if alt is None:
        if not segment.state.initials: raise AttributeError('altitude not set')
        alt = -1.0 *segment.state.initials.conditions.frames.inertial.position_vector[-1,2]   

    if v0  is None: 
        v0 = np.linalg.norm(segment.state.initials.conditions.frames.inertial.velocity_vector[-1])
        
    # avoid having zero velocity since aero and propulsion models need non-zero Reynolds number
    if v0 == 0.0: v0 = 0.01
    if vf == 0.0: vf = 0.01
    
    # intial and final speed cannot be the same
    if v0 == vf:
        vf = vf + 0.01
        
    # repack
    segment.air_speed_start = v0
    segment.air_speed_end   = vf
    
    # pack conditions  
    segment.state.conditions.propulsion.throttle[:,0] = throttle  
    segment.state.conditions.freestream.altitude[:,0] = alt
    segment.state.conditions.frames.inertial.position_vector[:,2] = -alt # z points down    
    
# ----------------------------------------------------------------------
#  Solve Residuals
# ----------------------------------------------------------------------    

## @ingroup Methods-Missions-Segments-Cruise
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
    vf = segment.air_speed_end
    v  = conditions.frames.inertial.velocity_vector
    m  = conditions.weights.total_mass
    a  = conditions.frames.inertial.acceleration_vector

    segment.state.residuals.forces[:,0] = FT[:,0]/m[:,0] - a[:,0]
    segment.state.residuals.forces[:,1] = FT[:,2]/m[:,0] 
    segment.state.residuals.final_velocity_error = (v[-1,0] - vf)

    return