## @ingroup Methods-Flight_Dynamics-Dynamic_Stability
# compute_aero_derivatives.py
# 
# Created:   Aug 2021, R. Erhard
# Modified: 
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Data

import numpy as np 
from copy import deepcopy

## @ingroup Methods-Flight_Dynamics-Dynamic_Stability
def compute_aero_derivatives(segment, h=1e-6): 
    """This function takes an aircraft with analyses setup and initial state,
    and computes the aerodynamic derivatives about the initial state.
    
    Assumptions:
       Linerarized equations are used for each state variable

    Source:
      N/A

    Inputs:
      vehicle                SUAVE Vehicle 
      state                  Initial state of vehicle (after running mission segment of interest)

    Outputs: 
       aero_derivatives      Aerodynamic derivatives wrt perturbed state variables

    Properties Used:
       N/A
     """
    
    # extract states from converged mission segment
    
    throttle           = segment.state.conditions.propulsion.throttle
    velocity           = segment.state.conditions.freestream.velocity
    orientation_vector = segment.state.conditions.frames.body.inertial_rotations
    
    alpha           = orientation_vector[:,1]
    beta            = orientation_vector[:,2]
    
    # ----------------------------------------------------------------------------
    # Perturb each state variable
    # ----------------------------------------------------------------------------
    # deepcopy segment
    perturbed_segment = deepcopy(segment)
    
    # Alpha perturbation
    alpha_plus = alpha*(1+h)
    perturbed_segment.state.conditions.frames.body.inertial_rotations[:,1] = alpha_plus
    
    iterate = perturbed_segment.process.iterate
    iterate.conditions(perturbed_segment) 
    
    # set segment derivatives based on perturbed segment
    dCL    = perturbed_segment.state.conditions.aerodynamics.lift_coefficient - segment.state.conditions.aerodynamics.lift_coefficient
    dCD    = perturbed_segment.state.conditions.aerodynamics.drag_coefficient - segment.state.conditions.aerodynamics.drag_coefficient
    
    dAlpha = np.atleast_2d(alpha_plus-alpha).T
    
    dCL_dAlpha = dCL/dAlpha
    dCD_dAlpha = dCD/dAlpha
    
    segment.state.conditions.aero_derivatives.dCL_dAlpha = dCL_dAlpha
    segment.state.conditions.aero_derivatives.dCD_dAlpha = dCD_dAlpha
    
    
    # ----------------------------------------------------------------------------    
    # Throttle perturbation
    perturbed_segment = deepcopy(segment)
    throttle_plus   = throttle*(1+h)
    
    perturbed_segment.state.conditions.propulsion.throttle = throttle_plus
    iterate = perturbed_segment.process.iterate
    results_Throttle_plus = iterate.conditions(perturbed_segment) 

    ## ----------------------------------------------------------------------------     
    ## Velocity perturbation
    #perturbed_segment = deepcopy(segment)
    #velocity_plus   = velocity*(1+h)
    
    #perturbed_segment.state.conditions.freestream.velocity = velocity_plus
    #perturbed_segment.state.conditions.frames.inertial.velocity_vector[:,0] = velocity_plus[0][0]
    #results_Velocity_plus  = iterate.conditions() 

    # ---------------------------------------------------------------------------- 
    # Compute and store aerodynamic derivatives
    # ---------------------------------------------------------------------------- 
    aero_derivatives = Data()
    aero_derivatives.dCL_dAlpha   = (results_Alpha_plus.CL  - segment.state.conditions.aerodynamics.lift_coefficient)/(alpha*h)
    #aero_derivatives.dCD_dAlpha   = (results_Alpha_plus.CD - results_Alpha_minus.CD)/h  
    #aero_derivatives.dCT_dAlpha   = (results_Alpha_plus.CT  - results_Alpha_minus.CT)/h  
    #aero_derivatives.dMyaw_dAlpha = (results_Alpha_plus.Myaw - results_Alpha_minus.Myaw)/h 

    #aero_derivatives.dCL_dThrottle   = (results_Throttle_plus.CL  - results_Throttle_minus.CL)/h
    #aero_derivatives.dCD_dThrottle   = (results_Throttle_plus.CD - results_Throttle_minus.CD)/h  
    #aero_derivatives.dCT_dThrottle   = (results_Throttle_plus.CT  - results_Throttle_minus.CT)/h  
    #aero_derivatives.dMyaw_dThrottle = (results_Throttle_plus.Myaw  - results_Throttle_minus.Myaw)/h       
    
    #aero_derivatives.dCL_dV   = (results_Velocity_plus.CL  - results_Velocity_minus.CL)/h
    #aero_derivatives.dCD_dV   = (results_Velocity_plus.CD - results_Velocity_minus.CD)/h 
    #aero_derivatives.dCT_dV   = (results_Velocity_plus.CT  - results_Velocity_minus.CT)/h
    #aero_derivatives.dMyaw_dV = (results_Velocity_plus.Myaw - results_Velocity_minus.Myaw)/h
    
    return aero_derivatives



def evaluate_aero_results(analyses,vehicle,state):
    # run new single-point analysis with perturbed parameters:
    
    new_mission = single_point_mission(analyses,vehicle,state)
    
    results = new_mission.evaluate() 
    
    props        = vehicle.networks.battery_propeller.propellers
    prop_keys    = props.keys()
    props_r      = results.segments.single_point.conditions.noise.sources.propellers
    
    # store results
    Results     = Data()
    Results.CL  = results.segments.single_point.conditions.aerodynamics.lift_coefficient[0][0]
    Results.CD  = results.segments.single_point.conditions.aerodynamics.drag_coefficient[0][0]
    
    # initialize results for propeller parameters
    Results.CT   = np.zeros(len(props))
    Results.Myaw = np.zeros(len(props))
    
    for i,prop in enumerate(props_r):
        outputs = results.segments.single_point.state.conditions.noise.sources.propellers[list(prop_keys)[i]]
        
        Results.CT[i]   = prop.thrust_coefficient[0][0]
        Results.Myaw[i] = outputs.thrust_per_blade[0][0] * props[list(prop_keys)[i]].origin[0][1]

    
    return Results


def single_point_mission(analyses,vehicle,state):
    # Extract single point values from provided state
    body_angle     = state.conditions.aerodynamics.angle_of_attack
    throttle       = state.conditions.propulsion.throttle
    air_speed      = state.conditions.freestream.velocity  
    altitude       = state.conditions.freestream.altitude
    battery_energy = state.conditions.propulsion.battery_energy[-1]
    
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------
    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'mission'

    # airport
    airport = SUAVE.Attributes.Airports.Airport()
    airport.altitude   =  0. * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()

    mission.airport = airport    

    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments 
    
    # base segment
    base_segment = Segments.Segment()
    base_segment.process.iterate.initials.initialize_battery = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery
    base_segment.process.iterate.conditions.planet_position  = SUAVE.Methods.skip
    base_segment.state.numerics.number_control_points        = state.numerics.number_control_points

    
    # ------------------------------------------------------------------
    #  Single Point Segment: constant speed, altitude, angle, and throttle from reference state
    # ------------------------------------------------------------------ 
    segment = Segments.Single_Point.Fixed_Conditions(base_segment)
    segment.tag = "single_point" 
    segment.analyses.extend(analyses.base) 
    segment.battery_energy = battery_energy
    segment.altitude       =  altitude   
    segment.air_speed      =  air_speed  
    segment.body_angle     =  body_angle 
    segment.throttle       =  throttle  
    segment = vehicle.networks.battery_propeller.add_unknowns_and_residuals_to_segment(segment)  

    # add to misison
    mission.append_segment(segment)           
    
    return mission
    

  