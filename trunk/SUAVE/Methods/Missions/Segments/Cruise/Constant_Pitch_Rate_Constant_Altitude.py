## @ingroup Methods-Missions-Segments-Cruise
# Constant_Pitch_Rate_Constant_Altitude.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero
#           May 2019, T. MacDonald
#           Mar 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Unpack Unknowns
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Cruise
def initialize_conditions(segment):
    """Sets the specified conditions which are given for the segment type.

    Assumptions:
    Constant acceleration and constant altitude

    Source:
    N/A

    Inputs:
    segment.altitude                [meters]
    segment.pitch_initial           [radians]
    segment.pitch_final             [radians]
    segment.pitch_rate              [radians/second]

    Outputs:
    conditions.frames.body.inertial_rotations   [radians/second]
    conditions.frames.inertial.position_vector  [meters]
    conditions.freestream.altitude              [meters]
    conditions.frames.inertial.time             [seconds]

    Properties Used:
    N/A
    """       
    
    # unpack
    alt        = segment.altitude 
    T0         = segment.pitch_initial
    Tf         = segment.pitch_final 
    theta_dot  = segment.pitch_rate   
    conditions = segment.state.conditions 
    state      = segment.state
    
    # check for initial altitude
    if alt is None:
        if not segment.state.initials: raise AttributeError('altitude not set')
        alt = -1.0 * segment.state.initials.conditions.frames.inertial.position_vector[-1,2]
        
    # check for initial pitch
    if T0 is None:
        T0  =  state.initials.conditions.frames.body.inertial_rotations[-1,1]
        segment.pitch_initial = T0
    
    # dimensionalize time
    t_initial = conditions.frames.inertial.time[0,0]
    t_final   = (Tf-T0)/theta_dot + t_initial
    t_nondim  = state.numerics.dimensionless.control_points
    time      = t_nondim * (t_final-t_initial) + t_initial
    
    # set the body angle
    body_angle = theta_dot*time + T0
    segment.state.conditions.frames.body.inertial_rotations[:,1] = body_angle[:,0]    
    
    # pack
    segment.state.conditions.freestream.altitude[:,0]             = alt
    segment.state.conditions.frames.inertial.position_vector[:,2] = -alt # z points down
    segment.state.conditions.frames.inertial.time[:,0]            = time[:,0]
    
    
## @ingroup Methods-Missions-Segments-Cruise    
def residual_total_forces(segment):
    """ Calculates a residual based on forces
    
        Assumptions:
        The vehicle is accelerating, doesn't use gravity
        
        Inputs:
        state.conditions:
            frames.inertial.total_force_vector [Newtons]
            weights.total_mass                 [kg]
            frames.inertial.velocity_vector    [meters/second]
            
        Outputs:
        state:
            residuals.forces [meters/second^2]
            conditions.frames.inertial.acceleration_vector [meters/second^2]

        Properties Used:
        N/A
                                
    """       
    
    FT = segment.state.conditions.frames.inertial.total_force_vector
    m  = segment.state.conditions.weights.total_mass  
    v  = segment.state.conditions.frames.inertial.velocity_vector
    D  = segment.state.numerics.time.differentiate  
    
    # process and pack
    acceleration = np.dot(D,v)
    segment.state.conditions.frames.inertial.acceleration_vector = acceleration
    a  = segment.state.conditions.frames.inertial.acceleration_vector
    
    # horizontal
    segment.state.residuals.forces[:,0] = FT[:,0]/m[:,0] - a[:,0]
    
    # vertical
    segment.state.residuals.forces[:,1] = FT[:,2]  - a[:,2]

    return
## @ingroup Methods-Missions-Segments-Cruise
def unpack_unknowns(segment):
    """ Unpacks the throttle setting and velocity from the solver to the mission
    
        Assumptions:
        N/A
        
        Inputs:
            state.unknowns:
                throttle    [Unitless]
                velocity    [meters/second]
            
        Outputs:
            state.conditions:
                propulsion.throttle             [Unitless]
                frames.inertial.velocity_vector [meters/second]

        Properties Used:
        N/A
                                
    """       
    
    # unpack unknowns
    throttle  = segment.state.unknowns.throttle
    air_speed = segment.state.unknowns.velocity
    
    # apply unknowns
    segment.state.conditions.propulsion.throttle[:,0]             = throttle[:,0]
    segment.state.conditions.frames.inertial.velocity_vector[:,0] = air_speed[:,0]
    
    