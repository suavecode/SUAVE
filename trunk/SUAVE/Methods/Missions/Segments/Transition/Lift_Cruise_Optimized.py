## @ingroup Methods-Missions-Segments-Transition
# Lift_Cruise_Optimized.py
# 
# Created:  Apr 2018, M. Clarke
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np
from SUAVE.Core import Units, Data
from SUAVE.Components.Energy.Networks.Lift_Cruise import Lift_Cruise
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_mass
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
    conditions = segment.state.conditions
    throttle       = segment.state.unknowns.throttle                        
    throttle_lift  = segment.state.unknowns.throttle_lift                   
    Cp_prop        = segment.state.unknowns.propeller_power_coefficient     
    Cp_prop_lift   = segment.state.unknowns.propeller_power_coefficient_lift
    Voltage        = segment.state.unknowns.battery_voltage_under_load
    alt            = segment.altitude   
    
    # pack  
    conditions.propulsion.throttle                         = throttle[:,0]     
    conditions.propulsion.throttle_lift                    = throttle_lift[:,0]
    conditions.propulsion.propeller_power_coefficient      = Cp_prop[:,0]      
    conditions.propulsion.propeller_power_coefficient_lift = Cp_prop_lift[:,0]   
    conditions.propulsion.battery_voltage_under_load       = Voltage[:,0] 
    
    conditions.frames.inertial.position_vector[:,2]        = -alt # z points down
    conditions.freestream.altitude[:,0]                    =  alt # positive altitude in this context      

## @ingroup Methods-Missions-Segments-Climb   
def update_differentials(segment):
    """ On each iteration creates the differentials and integration funcitons from knowns about the problem. Sets the time at each point. Must return in dimensional time, with t[0] = 0. This is different from the common method as it also includes the scaling of operators.

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
    initials   = segment.state.initials
    x          = numerics.dimensionless.control_points
    D          = numerics.dimensionless.differentiate
    I          = numerics.dimensionless.integrate    
    v0         = segment.air_speed_start
    vf         = segment.air_speed_end  
    ax         = segment.acceleration   
    T0         = segment.pitch_initial
    Tf         = segment.pitch_final     
    
    dv = vf - v0
    acc = -ax  # maintain column array

    # get overall time step
    dt = -(dv/acc)

    # rescale operators
    x = x * dt
    D = D / dt
    I = I * dt
    
    # dimensionalize time
    t_initial = segment.state.conditions.frames.inertial.time[0,0]
    t_final   = (vf-v0)/ax # dt
    t_nondim  = segment.state.numerics.dimensionless.control_points
    time      = t_nondim * (t_final-t_initial)
    
    # Figure out vx
    vx = v0+time*ax
    
    # set the body angle
    if Tf > T0:
        body_angle = time*(Tf-T0)/(t_final-t_initial)
    else:
        body_angle = T0 - time*(T0-Tf)/(t_final-t_initial)
    segment.state.conditions.frames.body.inertial_rotations[:,1] = body_angle[:,0]     
       
    # pack  
    t_initial                                       = conditions.frames.inertial.time[0,0]
    numerics.time.control_points                    = x
    numerics.time.differentiate                     = D
    numerics.time.integrate                         = I
    conditions.frames.inertial.time[:,0]            = time[:,0]
    conditions.frames.inertial.velocity_vector[:,0] = vx[:,0]

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
def solve_constant_speed_constant_altitude_loiter(segment):
    
    """ The sets up an solves a mini segment that is a linear speed constant rate segment. The results become the initial conditions for an optimized climb segment later

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    segment.altitude       
    segment.air_speed        
    segment.distance         
    conditions.frames.inertial.time [seconds]

    Outputs:
    conditions.frames.inertial.velocity_vector  [meters/second]
    conditions.frames.inertial.position_vector  [meters]
    conditions.freestream.altitude              [meters]
    conditions.frames.inertial.time             [seconds]

    Properties Used:
    N/A
    """ 
    net = Lift_Cruise()
    net.voltage = 400.     
    
    mini_mission = SUAVE.Analyses.Mission.Sequential_Segments()
    
    CACPCA = SUAVE.Analyses.Mission.Segments.Transition.Constant_Acceleration_Constant_Pitchrate_Constant_Altitude()   
    ones_row  = segment.state.ones_row  
    
    CACPCA.time             = segment.time
    CACPCA.analyses         = segment.analyses
    CACPCA.state.conditions = segment.state.conditions
    CACPCA.state.numerics   = segment.state.numerics      
    
    CACPCA.tag             = "transition"
    CACPCA.altitude        = 40.  * Units.ft
    CACPCA.air_speed_start = 0.   * Units['ft/min']
    CACPCA.air_speed_end   = 45
    CACPCA.acceleration    = 9.81/5
    CACPCA.pitch_initial   = 0.0
    CACPCA.pitch_final     = 7.75 * Units.degrees
    
    CACPCA.state.unknowns                              =  segment.state.unknowns   
    CACPCA.state.residuals.forces                      = 0.0  * ones_row(2) 
    CACPCA.state.residuals.network                     = 0.   * ones_row(3)  

    CACPCA.process.iterate.unknowns.network          = net.unpack_unknowns_transition
    CACPCA.process.iterate.residuals.network         = net.residuals_transition    
    CACPCA.process.iterate.unknowns.mission          = SUAVE.Methods.skip
    CACPCA.process.iterate.conditions.stability      = SUAVE.Methods.skip
    CACPCA.process.finalize.post_process.stability   = SUAVE.Methods.skip    
    
    mini_mission.append_segment(CACPCA)
        
    results  = mini_mission.evaluate()
    CACPCA_res = results.segments.analysis
     
    segment.state.unknowns.throttle                         = CACPCA_res.state.unknowns.throttle   
    segment.state.unknowns.throttle_lift                    = CACPCA_res.state.unknowns.throttle_lift
    segment.state.unknowns.propeller_power_coefficient      = CACPCA_res.state.unknowns.propeller_power_coefficient       
    segment.state.unknowns.propeller_power_coefficient_lift = CACPCA_res.state.unknowns.propeller_power_coefficient_lift

