## @ingroup Methods-Missions-Segments-Ground
# Common.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero
#           Mar 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
import SUAVE
from SUAVE.Methods.Geometry.Three_Dimensional \
     import angles_to_dcms, orientation_product, orientation_transpose

# ----------------------------------------------------------------------
#  Unpack Unknowns
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Ground
def unpack_unknowns(segment):
    """ Unpacks the times and velocities from the solver to the mission
    
        Assumptions:
        Overrides the velocities if they go to zero
        
        Inputs:
            segment.state.unknowns:
                velocity_x         [meters/second]
                time               [second]
            segment.velocity_start [meters/second]
            
        Outputs:
            segment.state.conditions:
                frames.inertial.velocity_vector [meters/second]
                frames.inertial.time            [second]
        Properties Used:
        N/A
                                
    """       
    
    # unpack unknowns
    unknowns   = segment.state.unknowns
    velocity_x = unknowns.velocity_x
    time       = unknowns.time
    
    # unpack givens
    v0         = segment.air_speed_start  
    t_initial  = segment.state.conditions.frames.inertial.time[0,0]
    t_nondim   = segment.state.numerics.dimensionless.control_points
    
    # time
    t_final    = t_initial + time  
    times      = t_nondim * (t_final-t_initial) + t_initial     

    #apply unknowns
    conditions = segment.state.conditions
    conditions.frames.inertial.velocity_vector[1:,0] = velocity_x
    conditions.frames.inertial.velocity_vector[0,0]  = v0
    conditions.frames.inertial.time[:,0]             = times[:,0]
     
# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Ground
def initialize_conditions(segment):
    """Sets the specified conditions which are given for the segment type.
    
    Assumptions:
       Checks to make sure non of the velocities are exactly zero
    
    Source:
       N/A
    
    Inputs:
       segment.velocity_start             [meters]
       segment.velocity_end               [meters]
       segment.speed                      [meters/second]
       segment.friction_coefficient       [unitless]
       segment.ground_incline             [radians]
    
    Outputs:
       conditions.frames.inertial.velocity_vector  [meters/second]
       conditions.ground.incline                   [radians]
       conditions.ground.friction_coefficient      [unitless]
       state.unknowns.velocity_x                   [meters/second]
    
    Properties Used:
       N/A
    """   

    # unpack inputs
    alt      = segment.altitude 
    v0       = segment.velocity_start
    vf       = segment.velocity_end 
    throttle = segment.throttle	
    
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
    
    initialized_velocity = (vf - v0)*segment.state.numerics.dimensionless.control_points + v0
    
    # Initialize the x velocity unknowns to speed convergence:
    segment.state.unknowns.velocity_x = initialized_velocity[1:,0]    

    # pack conditions 
    segment.state.conditions.frames.inertial.velocity_vector[:,0] = initialized_velocity[:,0]
    segment.state.conditions.ground.incline[:,0]                  = segment.ground_incline
    segment.state.conditions.ground.friction_coefficient[:,0]     = segment.friction_coefficient
    segment.state.conditions.propulsion.throttle[:,0]             = throttle  
    segment.state.conditions.freestream.altitude[:,0]             = alt
    segment.state.conditions.frames.inertial.position_vector[:,2] = -alt # z points down        
    
# ----------------------------------------------------------------------
#  Compute Ground Forces
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Ground
def compute_ground_forces(segment):
    """ Compute the rolling friction on the aircraft 
    
    Assumptions:
        Does a force balance to calculate the load on the wheels using only lift. Uses only a single friction coefficient.
    
    Source:
        N/A
    
    Inputs:
    conditions:
        frames.inertial.gravity_force_vector       [meters/second^2]
        ground.friction_coefficient                [unitless]
        frames.wind.lift_force_vector              [newtons]
    
    Outputs:
        conditions.frames.inertial.ground_force_vector [newtons]
    
    Properties Used:
        N/A
    """   

    # unpack
    conditions             = segment.state.conditions
    W                      = conditions.frames.inertial.gravity_force_vector[:,2,None]
    friction_coeff         = conditions.ground.friction_coefficient
    wind_lift_force_vector = conditions.frames.wind.lift_force_vector

    #transformation matrix to get lift in inertial frame
    T_wind2inertial = conditions.frames.wind.transform_to_inertial

    # to inertial frame
    L = orientation_product(T_wind2inertial,wind_lift_force_vector)[:,2,None]

    #compute friction force
    N  = -(W + L)
    Ff = N * friction_coeff

    #pack results. Friction acts along x-direction
    conditions.frames.inertial.ground_force_vector[:,2] = N[:,0]
    conditions.frames.inertial.ground_force_vector[:,0] = Ff[:,0]

# ----------------------------------------------------------------------
#  Compute Forces
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Ground
def compute_forces(segment):
    """ Adds the rolling friction to the traditional 4 forces of flight
    
    Assumptions:
    
    Source:
    N/A
    
    Inputs:
    conditions:
        frames.inertial.total_force_vector  [newtons]
        frames.inertial.ground_force_vector [newtons]
    
    Outputs:
        frames.inertial.ground_force_vector     [newtons]
    
    Properties Used:
        N/A
    """       


    SUAVE.Methods.Missions.Segments.Common.Frames.update_forces(segment)

    # unpack forces
    conditions                   = segment.state.conditions
    total_aero_forces            = conditions.frames.inertial.total_force_vector
    inertial_ground_force_vector = conditions.frames.inertial.ground_force_vector

    # sum of the forces, including friction force
    F = total_aero_forces + inertial_ground_force_vector

    # pack
    conditions.frames.inertial.total_force_vector[:,:] = F[:,:]

# ----------------------------------------------------------------------
#  Solve Residual
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Ground
def solve_residuals(segment):
    """ Calculates a residual based on forces
    
        Assumptions:
        
        Inputs:
            segment.state.conditions:
                frames.inertial.total_force_vector    [Newtons]
                frames.inertial.velocity_vector       [meters/second]
                weights.total_mass                    [kg]
            segment.state.numerics.time.differentiate [vector]
            segment.velocity_end                      [meters/second]
            
        Outputs:
            segment.state:
                residuals.acceleration_x           [meters/second^2]
                residuals.final_velocity_error     [meters/second]
        Properties Used:
        N/A
                                
    """   

    # unpack inputs
    conditions = segment.state.conditions
    FT = conditions.frames.inertial.total_force_vector
    vf = segment.velocity_end
    v  = conditions.frames.inertial.velocity_vector
    m  = conditions.weights.total_mass
    D  = segment.state.numerics.time.differentiate

    # process and pack
    acceleration                                   = np.dot(D , v)
    conditions.frames.inertial.acceleration_vector = acceleration
    
    if vf == 0.0: vf = 0.01

    segment.state.residuals.forces[:,0]          = FT[1:,0]/m[1:,0] - acceleration[1:,0]
    segment.state.residuals.final_velocity_error = (v[-1,0] - vf)
    
    return