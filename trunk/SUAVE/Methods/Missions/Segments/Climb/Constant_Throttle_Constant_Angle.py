## @ingroup Methods-Missions-Segments-Cruise
# Constant_Throttle_Constant_Altitude.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Methods.Geometry.Three_Dimensional \
     import angles_to_dcms, orientation_product, orientation_transpose



# ----------------------------------------------------------------------
#  Unpack Unknowns
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Cruise
def unpack_unknowns(segment,state):
    
    # unpack unknowns
    unknowns        = state.unknowns
    velocity        = unknowns.velocity      #magnitude
    alts            = unknowns.altitudes   
    theta           = unknowns.body_angle
    
    # unpack inputs
    v0              = segment.air_speed_start
    climb_angle     = segment.climb_angle
    alt0            = segment.altitude_start
    altf            = segment.altitude_end
    throttle        = segment.throttle
    
    # unpack givens
    t_initial  = state.conditions.frames.inertial.time[0,0]
    t_nondim   = state.numerics.dimensionless.control_points
    

    # Calculations
    v_x   = velocity * np.cos(climb_angle)
    v_z   = -velocity * np.sin(climb_angle)
    
    v0_x  = v0*np.cos(climb_angle)
    v0_z  = -v0*np.sin(climb_angle)
    
    
    # pack conditions    
    #apply unknowns
    conditions = state.conditions
    
    # check for initial altitude
    if alt0 is None:
        if not state.initials: raise AttributeError('initial altitude not set')
        alt0 = -1.0 * state.initials.conditions.frames.inertial.position_vector[-1,2]
    
    # pack conditions    
    conditions.freestream.altitude[:,0]             =  alts[:,0] # positive altitude in this context    

        #-- pack velocities
            #---- X
    conditions.frames.inertial.velocity_vector[:,0] = v_x[:,0]
    conditions.frames.inertial.velocity_vector[0,0] = v0_x
            #---- Z
    conditions.frames.inertial.velocity_vector[:,2] = v_z[:,0]
    conditions.frames.inertial.velocity_vector[0,2] = v0_z
    
        #-- pack position
    conditions.frames.inertial.position_vector[:,2] = -alts[:,0] # z points down

  

    conditions.frames.body.inertial_rotations[:,1]  = theta[:,0]      
    conditions.propulsion.throttle[:,0] = throttle
# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------    


## @ingroup Methods-Missions-Segments-Cruise
def initialize_conditions(segment,state):
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
    


    conditions = state.conditions

    # unpack inputs
    climb_angle = segment.climb_angle
    throttle    = segment.throttle
    alt0        = segment.altitude_start
    altf        = segment.altitude_end
    v0          = segment.air_speed_start
    N           = segment.state.numerics.number_control_points   
    
    # check for initial altitude
    if alt0 is None:
        if not state.initials: raise AttributeError('altitude not set')
        alt = -1.0 * state.initials.conditions.frames.inertial.position_vector[-1,2]
        segment.altitude = alt    

    # avoid having zero velocity since aero and propulsion models need non-zero Reynolds number
    if v0 == 0.0: v0 = 0.01

    # repack
    segment.air_speed_start = v0
    
    # pack conditions
    state.conditions.propulsion.throttle[:,0] = throttle  

# ----------------------------------------------------------------------
#  Solve Residuals
# ----------------------------------------------------------------------    

## @ingroup Methods-Missions-Segments-Cruise
def solve_residuals(segment,state):
    """ Calculates a residual based on forces
    
        Assumptions:
        The vehicle accelerates, residual on forces and to get it to the final speed
        
        Inputs:
        segment.air_speed_end                  [meters/second]
        state.conditions:
            frames.inertial.total_force_vector [Newtons]
            frames.inertial.velocity_vector    [meters/second]
            weights.total_mass                 [kg]
        state.numerics.time.differentiate
            
        Outputs:
        state.residuals:
            forces               [meters/second^2]
            final_velocity_error [meters/second]
        state.conditions:
            conditions.frames.inertial.acceleration_vector [meters/second^2]

        Properties Used:
        N/A
                                
    """    

    # unpack inputs
    conditions = state.conditions
    FT = conditions.frames.inertial.total_force_vector
    v  = conditions.frames.inertial.velocity_vector
    D  = state.numerics.time.differentiate
    m  = conditions.weights.total_mass
    
    alt_in  = state.unknowns.altitudes[:,0] 
    alt_out = state.conditions.freestream.altitude[:,0] 
    
    # process and pack
    acceleration = np.dot(D , v)
    conditions.frames.inertial.acceleration_vector = acceleration
    
    a  = state.conditions.frames.inertial.acceleration_vector

    state.residuals.forces[:,0] = FT[:,0]/m[:,0] - a[:,0]
    state.residuals.forces[:,1] = FT[:,2]/m[:,0] - a[:,2]   
    state.residuals.forces[:,2] = (alt_in - alt_out)/alt_out[-1]

    return











##############################################################3
##################################################3
###############################################
##########################################3
######################################
###################################
##############################3
#########################
########################
######################
###################
################
##############
############
#########
#######3
#


## ----------------------------------------------------------------------
##  Unpack Unknowns
## ----------------------------------------------------------------------
#
### @ingroup Methods-Missions-Segments-Cruise
#def unpack_unknowns(segment,state):
#    
#    # unpack unknowns
#    unknowns   = state.unknowns
#    velocity   = unknowns.velocity #magnitude
#    time       = unknowns.time
#    theta      = unknowns.body_angle
#    alts       = unknowns.altitudes  
#    print np.shape(velocity)
#
#    # unpack givens
#    v0         = segment.air_speed_start  
#
#    t_initial  = state.conditions.frames.inertial.time[0,0]
#    t_nondim   = state.numerics.dimensionless.control_points
#    
#    v0          = segment.air_speed_start
#    alt0        = segment.altitude_start
#    altf        = segment.altitude_end
#    climb_angle = segment.climb_angle
#    
#    # time
#    t_final    = t_initial + time  
#    time       = t_nondim * (t_final-t_initial) + t_initial     
#    
#
#    v_x   = velocity * np.cos(climb_angle)
#    v_z   = -velocity * np.sin(climb_angle)
#    v0_x  = v0*np.cos(climb_angle)
#    v0_z  = -v0*np.sin(climb_angle)
#    # pack conditions    
#    #apply unknowns
#    conditions = state.conditions
#    
#    conditions.frames.inertial.velocity_vector[:,0] = v_x[:,0]
#    conditions.frames.inertial.velocity_vector[:,2] = v_z[:,0]
#
#
#    conditions.frames.inertial.velocity_vector[0,0] = v0_x
#    conditions.frames.inertial.velocity_vector[0,2] = v0_z
#    conditions.frames.body.inertial_rotations[:,1]  = theta[:,0]  
#    conditions.frames.inertial.position_vector[:,2] = -alts[:,0] # z points down    
#    conditions.frames.inertial.time[:,0]            = time[:,0]
#    conditions.frames.body.inertial_rotations[:,1]  = theta[:,0]      
#    
## ----------------------------------------------------------------------
##  Initialize Conditions
## ----------------------------------------------------------------------    
#
#
#
#
#
#
#
#
### @ingroup Methods-Missions-Segments-Cruise
#def initialize_conditions(segment,state):
#    """Sets the specified conditions which are given for the segment type.
#
#    Assumptions:
#    Constant throttle and constant altitude, allows for acceleration
#
#    Source:
#    N/A
#
#    Inputs:
#    segment.altitude                             [meters]
#    segment.air_speed_start                      [meters/second]
#    segment.air_speed_end                        [meters/second]
#    segment.throttle	                         [unitless]
#    segment.state.numerics.number_control_points [int]
#
#    Outputs:
#    state.conditions.propulsion.throttle        [unitless]
#    conditions.frames.inertial.position_vector  [meters]
#    conditions.freestream.altitude              [meters]
#
#    Properties Used:
#    N/A
#    """   
#    
#
#
#    conditions = state.conditions
#
#    # unpack inputs
#    climb_angle = segment.climb_angle
#    throttle    = segment.throttle
#    alt0        = segment.altitude_start
#    altf        = segment.altitude_end
#    v0          = segment.air_speed_start
#    N           = segment.state.numerics.number_control_points   
#    
#    # check for initial altitude
#    if alt0 is None:
#        if not state.initials: raise AttributeError('altitude not set')
#        alt = -1.0 * state.initials.conditions.frames.inertial.position_vector[-1,2]
#        segment.altitude = alt    
#
#    # avoid having zero velocity since aero and propulsion models need non-zero Reynolds number
#    if v0 == 0.0: v0 = 0.01
#
#    # repack
#    segment.air_speed_start = v0
#    
#    # Initialize the x velocity unknowns to speed convergence:
#    state.unknowns.alts = np.linspace(alt0,altf,N)
#    
#    # pack conditions
#    state.conditions.propulsion.throttle[:,0] = throttle  
#
## ----------------------------------------------------------------------
##  Solve Residuals
## ----------------------------------------------------------------------    
#
### @ingroup Methods-Missions-Segments-Cruise
#def solve_residuals(segment,state):
#    """ Calculates a residual based on forces
#    
#        Assumptions:
#        The vehicle accelerates, residual on forces and to get it to the final speed
#        
#        Inputs:
#        segment.air_speed_end                  [meters/second]
#        state.conditions:
#            frames.inertial.total_force_vector [Newtons]
#            frames.inertial.velocity_vector    [meters/second]
#            weights.total_mass                 [kg]
#        state.numerics.time.differentiate
#            
#        Outputs:
#        state.residuals:
#            forces               [meters/second^2]
#            final_velocity_error [meters/second]
#        state.conditions:
#            conditions.frames.inertial.acceleration_vector [meters/second^2]
#
#        Properties Used:
#        N/A
#                                
#    """    
#
#    # unpack inputs
#    conditions = state.conditions
#    FT = conditions.frames.inertial.total_force_vector
#    altf = segment.altitude_end
#    v  = conditions.frames.inertial.velocity_vector
#    D  = state.numerics.time.differentiate
#    m  = conditions.weights.total_mass
#    r = conditions.frames.inertial.position_vector
#    # process and pack
#    acceleration = np.dot(D , v)
#    conditions.frames.inertial.acceleration_vector = acceleration
#    
#    a  = state.conditions.frames.inertial.acceleration_vector
#
#    state.residuals.forces[:,0] = FT[:,0]/m[:,0] - a[:,0]
#    state.residuals.forces[:,1] = FT[:,2]/m[:,0] - a[:,2]   
#    state.residuals.forces[:,2] = (-r[-1,2] - altf) #residual on altitude
#
#    return