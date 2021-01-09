## @ingroup Methods-Missions-Segments-Climb
# Optimized.py
# 
# Created:  Dec 2016, E. Botero
# Modified: Mar 2020, M. Clarke
#           Apr 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np
from SUAVE.Core import Units
import SUAVE

# ----------------------------------------------------------------------
#  Unpack Unknowns
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Climb
def unpack_unknowns(segment):
    
    """Unpacks the unknowns set in the mission to be available for the mission.

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    segment.state.unknowns.throttle            [Unitless]
    segment.state.unknowns.body_angle          [Radians]
    segment.state.unknowns.flight_path_angle   [Radians]
    segment.state.unknowns.velocity            [meters/second]
    segment.altitude_start                     [meters]
    segment.altitude_end                       [meters]
    segment.air_speed_start                    [meters/second]
    segment.air_speed_end                      [meters/second]

    Outputs:
    segment.state.conditions.propulsion.throttle            [Unitless]
    segment.state.conditions.frames.body.inertial_rotations [Radians]
    conditions.frames.inertial.velocity_vector              [meters/second]

    Properties Used:
    N/A
    """    
    
    # unpack unknowns and givens
    throttle = segment.state.unknowns.throttle
    theta    = segment.state.unknowns.body_angle
    gamma    = segment.state.unknowns.flight_path_angle
    vel      = segment.state.unknowns.velocity
    vel0     = segment.air_speed_start
    velf     = segment.air_speed_end

    # Overide the speeds   
    if segment.air_speed_end is None:
        v_mag =  np.concatenate([[[vel0]],vel])
    elif segment.air_speed_end is not None:
        v_mag = np.concatenate([[[vel0]],vel,[[velf]]])
        
    if np.all(gamma == 0.):
        gamma[gamma==0.] = 1.e-16
        
    if np.all(vel == 0.):
        vel[vel==0.] = 1.e-16
    
    # process velocity vector
    v_x   =  v_mag * np.cos(gamma)
    v_z   = -v_mag * np.sin(gamma)    

    # apply unknowns and pack conditions   
    segment.state.conditions.propulsion.throttle[:,0]             = throttle[:,0]
    segment.state.conditions.frames.body.inertial_rotations[:,1]  = theta[:,0]
    segment.state.conditions.frames.inertial.velocity_vector[:,0] = v_x[:,0]
    segment.state.conditions.frames.inertial.velocity_vector[:,2] = v_z[:,0]

## @ingroup Methods-Missions-Segments-Climb   
def update_differentials(segment):
    """ On each iteration creates the differentials and integration functions from knowns about the problem. Sets the time at each point. Must return in dimensional time, with t[0] = 0. This is different from the common method as it also includes the scaling of operators.

        Assumptions:
        Works with a segment discretized in vertical position, altitude

        Inputs:
        state.numerics.dimensionless.control_points      [Unitless]
        state.numerics.dimensionless.differentiate       [Unitless]
        state.numerics.dimensionless.integrate           [Unitless]
        state.conditions.frames.inertial.position_vector [meter]
        state.conditions.frames.inertial.velocity_vector [meter/second]
        

        Outputs:
        state.conditions.frames.inertial.time            [second]

    """    

    # unpack
    numerics   = segment.state.numerics
    conditions = segment.state.conditions
    x          = numerics.dimensionless.control_points
    D          = numerics.dimensionless.differentiate
    I          = numerics.dimensionless.integrate 
    r          = segment.state.conditions.frames.inertial.position_vector
    v          = segment.state.conditions.frames.inertial.velocity_vector
    alt0       = segment.altitude_start
    altf       = segment.altitude_end    

    dz = altf - alt0
    vz = -v[:,2,None] # maintain column array

    # get overall time step
    dt = (dz/np.dot(I,vz))[-1]

    # rescale operators
    x = x * dt
    D = D / dt
    I = I * dt
    
    # Calculate the altitudes
    alt = np.dot(I,vz) + segment.altitude_start
    
    # pack
    t_initial                                       = segment.state.conditions.frames.inertial.time[0,0]
    numerics.time.control_points                    = x
    numerics.time.differentiate                     = D
    numerics.time.integrate                         = I
    conditions.frames.inertial.time[1:,0]            = t_initial + x[1:,0]
    conditions.frames.inertial.position_vector[:,2] = -alt[:,0] # z points down
    conditions.freestream.altitude[:,0]             =  alt[:,0] # positive altitude in this context    

    return

## @ingroup Methods-Missions-Segments-Climb
def objective(segment):
    """ This function pulls the objective from the results of flying the segment and returns it to the optimizer
    
        Inputs:
        state
        
        Outputs:
        state.objective_value [float]

    """       
    
    
    # If you have an objective set, either maximize or minimize
    if segment.objective is not None:
        if segment.minimize ==True:
            objective = eval('segment.state.'+segment.objective)
        else:
            objective = -eval('segment.state.'+segment.objective)
    else:
        objective = 0.
    # No objective is just solved constraint like a normal mission    
        
    segment.state.objective_value = objective
        
## @ingroup Methods-Missions-Segments-Climb
def constraints(segment):
    """ This function pulls the equality constraints from the results of flying the segment and returns it to the optimizer

        Inputs:
        state
        
        Outputs:
        state.constraint_values [vector]

    """       
    
    # Residuals
    segment.state.constraint_values = segment.state.residuals.pack_array()
        
## @ingroup Methods-Missions-Segments-Climb
def cache_inputs(segment):
    """ This function caches the prior inputs to make sure the same inputs are not run twice in a row

    """      
    segment.state.inputs_last = segment.state.unknowns.pack_array()
    
## @ingroup Methods-Missions-Segments-Climb
def solve_linear_speed_constant_rate(segment):
    
    """ The sets up an solves a mini segment that is a linear speed constant rate segment. The results become the initial conditions for an optimized climb segment later

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    segment.altitude_start             [meters]
    segment.altitude_end               [meters]
    segment.air_speed_start            [meters/second]
    segment.air_speed_end              [meters/second]
    segment.analyses                   [Data]
    state.numerics                     [Data]

    Outputs:
    state.unknowns.throttle            [Unitless]
    state.unknowns.body_angle          [Radians]
    state.unknowns.flight_path_angle   [Radians]
    state.unknowns.velocity            [meters/second]
    
    Properties Used:
    N/A    
    
    """
    
    mini_mission = SUAVE.Analyses.Mission.Sequential_Segments()
    
    LSCR = SUAVE.Analyses.Mission.Segments.Climb.Linear_Speed_Constant_Rate()
    LSCR.air_speed_start = segment.air_speed_start
    
    if segment.air_speed_end is not None:
        LSCR.air_speed_end   = segment.air_speed_end
    else:
        LSCR.air_speed_end   = segment.air_speed_start
        
    LSCR.altitude_start   = segment.altitude_start
    LSCR.altitude_end     = segment.altitude_end
    LSCR.climb_rate       = segment.seed_climb_rate
    LSCR.analyses         = segment.analyses
    LSCR.state.conditions = segment.state.conditions
    LSCR.state.numerics   = segment.state.numerics
    mini_mission.append_segment(LSCR)
    
    results = mini_mission.evaluate()
    LSCR_res = results.segments.analysis
    
    segment.state.unknowns.body_angle        = LSCR_res.state.unknowns.body_angle
    segment.state.unknowns.throttle          = LSCR_res.state.unknowns.throttle
    segment.state.unknowns.flight_path_angle = LSCR_res.state.unknowns.body_angle - LSCR_res.conditions.aerodynamics.angle_of_attack
    
    # Make the velocity vector
    v_mag = np.linalg.norm(LSCR_res.state.conditions.frames.inertial.velocity_vector,axis=1)
    
    if segment.air_speed_end is None:
        segment.state.unknowns.velocity =  np.reshape(v_mag[1:],(-1, 1))
    elif segment.air_speed_end is not None:    
        segment.state.unknowns.velocity = np.reshape(v_mag[1:-1],(-1, 1))