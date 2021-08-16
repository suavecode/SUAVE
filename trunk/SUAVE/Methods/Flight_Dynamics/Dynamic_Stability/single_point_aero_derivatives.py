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
def single_point_aero_derivatives(vehicle, state, h=0.01): 
    """This function takes an aircraft with mission segment and computes
    the aerodynamic derivatives to be used for linearized aircraft models.
    
    Assumptions:
       Linerarized equations are used for each state variable
       Identical propellers

    Source:
      N/A

    Inputs:
      vehicle                SUAVE Vehicle 
      state                  State of vehicle

    Outputs: 
       aero_derivatives      Aerodynamic derivatives wrt state variables

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
    results_Alpha_plus = evaluate_aero_results(vehicle,perturbed_state)
    
    perturbed_state.conditions.aerodynamics.angle_of_attack = alpha_minus
    results_Alpha_minus = evaluate_aero_results(vehicle,perturbed_state)    

    # ----------------------------------------------------------------------------    
    # Throttle perturbation
    
    throttle_plus  = throttle + 0.5*h
    throttle_minus = throttle - 0.5*h
    
    perturbed_state = deepcopy(state)
    perturbed_state.conditions.propulsion.throttle = throttle_plus
    results_Throttle_plus = evaluate_aero_results(vehicle,perturbed_state)
    perturbed_state.conditions.propulsion.throttle = throttle_minus
    results_Throttle_minus = evaluate_aero_results(vehicle,perturbed_state)

    # ----------------------------------------------------------------------------     
    # Velocity perturbation
    perturbed_state = deepcopy(state)
    velocity_plus  = velocity + 0.5*h
    velocity_minus = velocity - 0.5*h
    
    perturbed_state.conditions.freestream.velocity = velocity_plus
    perturbed_state.conditions.frames.inertial.velocity_vector[:,0] = velocity_plus[0][0]
    results_Velocity_plus = evaluate_aero_results(vehicle,perturbed_state)
    
    perturbed_state.conditions.freestream.velocity = velocity_minus
    perturbed_state.conditions.frames.inertial.velocity_vector[:,0] = velocity_minus[0][0]
    results_Velocity_minus = evaluate_aero_results(vehicle,perturbed_state)        

    # ---------------------------------------------------------------------------- 
    # Compute and store aerodynamic derivatives
    # ---------------------------------------------------------------------------- 
    aero_derivatives = Data()
    aero_derivatives.dCL_dAlpha  = (results_Alpha_plus.CL  - results_Alpha_minus.CL)/h
    aero_derivatives.dCDi_dAlpha = (results_Alpha_plus.CDi - results_Alpha_minus.CDi)/h  
    aero_derivatives.dCM_dAlpha  = (results_Alpha_plus.CM - results_Alpha_minus.CM)/h  
    aero_derivatives.dT_dAlpha   = (results_Alpha_plus.T  - results_Alpha_minus.T)/h  
    aero_derivatives.dYaw_dAlpha = (results_Alpha_plus.Yaw - results_Alpha_minus.Yaw)/h 

    aero_derivatives.dCL_dThrottle   = (results_Throttle_plus.CL  - results_Throttle_minus.CL)/h
    aero_derivatives.dCDi_dThrottle  = (results_Throttle_plus.CDi - results_Throttle_minus.CDi)/h  
    aero_derivatives.dCM_dThrottle   = (results_Throttle_plus.CM - results_Throttle_minus.CM)/h  
    aero_derivatives.dT_dThrottle    = (results_Throttle_plus.T  - results_Throttle_minus.T)/h  
    aero_derivatives.dYaw_dThrottle  = (results_Throttle_plus.Yaw  - results_Throttle_minus.Yaw)/h       
    
    aero_derivatives.dCL_dV  = (results_Velocity_plus.CL  - results_Velocity_minus.CL)/h
    aero_derivatives.dCDi_dV = (results_Velocity_plus.CDi - results_Velocity_minus.CDi)/h 
    aero_derivatives.dCM_dV  = (results_Velocity_plus.CM - results_Velocity_minus.CM)/h
    aero_derivatives.dT_dV   = (results_Velocity_plus.T  - results_Velocity_minus.T)/h
    aero_derivatives.dYaw_dV = (results_Velocity_plus.Yaw - results_Velocity_minus.Yaw)/h
    
    return aero_derivatives 


def evaluate_aero_results(vehicle,state):

    # -------------------------------------------------------------------------------
    # run the VLM with given conditions
    # -------------------------------------------------------------------------------
    vlm_settings  = SUAVE.Analyses.Aerodynamics.Vortex_Lattice().settings
    vlm_results   = VLM(state.conditions,vlm_settings,vehicle)     
    
    # -------------------------------------------------------------------------------
    # run the BEMT
    # -------------------------------------------------------------------------------    
    props        = vehicle.networks.battery_propeller.propellers
    Myaw         = np.zeros(len(props))
    T            = np.zeros(len(props))
    
    # update thrust evaluation for new state
    vehicle.networks.battery_propeller.evaluate_thrust(state)
    
    # evaluate each propeller
    for i,prop in enumerate(props):
        
        # update rpm from new state
        prop.inputs.omega = state.conditions.propulsion.propeller_rpm
        
        # spin each propeller at given conditions
        F, Q, P, Cp ,  outputs , etap = prop.spin(state.conditions) 
        
        # orientation angle:
        #euler_angles = prop.orientation_euler_angles
        p_y          = prop.origin[0][1]
        
        # evaluate vehicle moments introduced by this propeller's forces, first control point
        T[i]           = F[0][0]
        Myaw[i]        = T[i]*p_y
    
    
    # store results
    Results     = Data()
    Results.CL  = vlm_results.CL[0][0]
    Results.CDi = vlm_results.CDi[0][0]
    Results.CM  = vlm_results.CM[0][0]
    Results.T   = T
    Results.Yaw = Myaw

    
    return Results