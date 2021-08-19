## @ingroup Methods-Flight_Dynamics-Dynamic_Stability
# single_point_aero_derivatives.py
# 
# Created:   Aug 2021, R. Erhard
# Modified: 
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Data
from SUAVE.Plots.Mission_Plots import *  
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.VLM import  VLM  

import numpy as np 
from copy import deepcopy

## @ingroup Methods-Flight_Dynamics-Dynamic_Stability
def single_point_aero_derivatives(analyses,vehicle, state, h=0.01): 
    """This function takes an aircraft with mission segment and computes
    the aerodynamic derivatives to be used for linearized aircraft models.
    
    Assumptions:
       Linerarized equations are used for each state variable
       Identical propellers

    Source:
      N/A

    Inputs:
      vehicle                SUAVE Vehicle 
      state                  Initial state of vehicle

    Outputs: 
       aero_derivatives      Aerodynamic derivatives wrt perturbed state variables

    Properties Used:
       N/A
     """
    
    # extract states from results for mission segment
    alpha    = state.conditions.aerodynamics.angle_of_attack
    throttle = state.conditions.propulsion.throttle
    velocity = state.conditions.freestream.velocity
    
    # ----------------------------------------------------------------------------
    # Perturb each state variable
    # ----------------------------------------------------------------------------
    # Alpha perturbation
    perturbed_state = deepcopy(state)
    alpha_plus  = alpha + 0.5*h
    alpha_minus = alpha - 0.5*h
    
    perturbed_state.conditions.aerodynamics.angle_of_attack = alpha_plus
    
    # test the single point set speed set throttle result
    
    
    results_Alpha_plus = evaluate_aero_results(analyses,vehicle,perturbed_state)
    
    perturbed_state.conditions.aerodynamics.angle_of_attack = alpha_minus
    results_Alpha_minus = evaluate_aero_results(analyses,vehicle,perturbed_state)    

    # ----------------------------------------------------------------------------    
    # Throttle perturbation
    
    throttle_plus  = throttle + 0.5*h
    throttle_minus = throttle - 0.5*h
    
    perturbed_state = deepcopy(state)
    perturbed_state.conditions.propulsion.throttle = throttle_plus
    results_Throttle_plus = evaluate_aero_results(analyses,vehicle,perturbed_state)
    perturbed_state.conditions.propulsion.throttle = throttle_minus
    results_Throttle_minus = evaluate_aero_results(analyses,vehicle,perturbed_state)

    # ----------------------------------------------------------------------------     
    # Velocity perturbation
    perturbed_state = deepcopy(state)
    velocity_plus  = velocity + 0.5*h
    velocity_minus = velocity - 0.5*h
    
    perturbed_state.conditions.freestream.velocity = velocity_plus
    perturbed_state.conditions.frames.inertial.velocity_vector[:,0] = velocity_plus[0][0]
    results_Velocity_plus = evaluate_aero_results(analyses,vehicle,perturbed_state)
    
    perturbed_state.conditions.freestream.velocity = velocity_minus
    perturbed_state.conditions.frames.inertial.velocity_vector[:,0] = velocity_minus[0][0]
    results_Velocity_minus = evaluate_aero_results(analyses,vehicle,perturbed_state)        

    # ---------------------------------------------------------------------------- 
    # Compute and store aerodynamic derivatives
    # ---------------------------------------------------------------------------- 
    aero_derivatives = Data()
    aero_derivatives.dCL_dAlpha  = (results_Alpha_plus.CL  - results_Alpha_minus.CL)/h
    aero_derivatives.dCD_dAlpha = (results_Alpha_plus.CD - results_Alpha_minus.CD)/h  
    aero_derivatives.dCM_dAlpha  = 0#(results_Alpha_plus.CM - results_Alpha_minus.CM)/h  
    #aero_derivatives.dT_dAlpha   = (results_Alpha_plus.T  - results_Alpha_minus.T)/h  
    #aero_derivatives.dYaw_dAlpha = (results_Alpha_plus.Yaw - results_Alpha_minus.Yaw)/h 

    aero_derivatives.dCL_dThrottle   = (results_Throttle_plus.CL  - results_Throttle_minus.CL)/h
    aero_derivatives.dCD_dThrottle  = (results_Throttle_plus.CD - results_Throttle_minus.CD)/h  
    aero_derivatives.dCM_dThrottle   = 0#(results_Throttle_plus.CM - results_Throttle_minus.CM)/h  
    #aero_derivatives.dT_dThrottle    = (results_Throttle_plus.T  - results_Throttle_minus.T)/h  
    #aero_derivatives.dYaw_dThrottle  = (results_Throttle_plus.Yaw  - results_Throttle_minus.Yaw)/h       
    
    aero_derivatives.dCL_dV  = (results_Velocity_plus.CL  - results_Velocity_minus.CL)/h
    aero_derivatives.dCD_dV = (results_Velocity_plus.CD - results_Velocity_minus.CD)/h 
    aero_derivatives.dCM_dV  = 0# (results_Velocity_plus.CM - results_Velocity_minus.CM)/h
    #aero_derivatives.dT_dV   = (results_Velocity_plus.T  - results_Velocity_minus.T)/h
    #aero_derivatives.dYaw_dV = (results_Velocity_plus.Yaw - results_Velocity_minus.Yaw)/h
    
    
    display_derivative_table(aero_derivatives)
    
    return aero_derivatives 

def display_derivative_table(aero_derivatives):
    
    # plot table of aero derivatives 
    column_labels=["$d\\alpha$", "dThrottle", "$dV_\\infty$"]
    row_labels=["$dC_L$", "$dC_{D}$", "$dC_M$"]
    
    plt.rcParams["figure.figsize"] = [7.00, 4.00]
    plt.rcParams["figure.autolayout"] = True    
    fig, ax =plt.subplots(1,1)
    
    data=np.array([[aero_derivatives.dCL_dAlpha,aero_derivatives.dCL_dThrottle,aero_derivatives.dCL_dV],
          [aero_derivatives.dCD_dAlpha,aero_derivatives.dCD_dThrottle,aero_derivatives.dCD_dV],
          [aero_derivatives.dCM_dAlpha,aero_derivatives.dCM_dThrottle,aero_derivatives.dCM_dV],])
    
    #for i in range(len(aero_derivatives.dT_dAlpha)):
        #data = np.append(data,[[aero_derivatives.dT_dAlpha[i],aero_derivatives.dT_dThrottle[i],aero_derivatives.dT_dV[i]]],axis=0)
        #row_labels = np.append(row_labels,f"$dT_{i}$")
        #data = np.append(data,[[aero_derivatives.dYaw_dAlpha[i],aero_derivatives.dYaw_dThrottle[i],aero_derivatives.dYaw_dV[i]]],axis=0)
        #row_labels = np.append(row_labels,f"$dMyaw_{i}$" )
    
    
    round_data = np.round(data,3)
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=round_data,rowLabels=row_labels,colLabels=column_labels,loc="center")  
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    plt.show()
    
    return

def evaluate_aero_results(analyses,vehicle,state):
    # run new single-point analysis with perturbed parameters:
    
    new_mission = single_point_mission(analyses,vehicle,state)
    
    results = new_mission.evaluate() 
    
    # store results
    Results     = Data()
    Results.CL  = results.segments.single_point.conditions.aerodynamics.lift_coefficient[0][0]
    Results.CD  = results.segments.single_point.conditions.aerodynamics.drag_coefficient[0][0]
    #Results.CM  = vlm_results.CM[0][0]
    Results.CT   = results.segments.single_point.conditions.noise.sources.propellers.propeller.thrust_coefficient[0][0]
    #Results.Yaw = Myaw    

    
    return Results


def single_point_mission(analyses,vehicle,state):
    # Extract single point values from state
    body_angle = state.conditions.aerodynamics.angle_of_attack
    throttle   = state.conditions.propulsion.throttle
    air_speed  = state.conditions.freestream.velocity  
    altitude   = state.conditions.freestream.altitude
    
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
    base_segment.state.numerics.number_control_points        = 4

    
    # ------------------------------------------------------------------
    #  Single Point Segment: constant speed, altitude, angle, and throttle from reference state
    # ------------------------------------------------------------------ 
    segment = Segments.Single_Point.Fixed_Conditions(base_segment)
    segment.tag = "single_point" 
    segment.analyses.extend(analyses.base) 
    segment.altitude    =  altitude   
    segment.air_speed   =  air_speed  
    segment.body_angle  =  body_angle 
    segment.throttle    =  throttle  
    segment = vehicle.networks.battery_propeller.add_unknowns_and_residuals_to_segment(segment)  

    # add to misison
    mission.append_segment(segment)           
    
    return mission
    

  