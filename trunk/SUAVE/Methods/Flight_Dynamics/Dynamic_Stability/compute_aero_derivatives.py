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
def compute_aero_derivatives(segment, h=1e-4): 
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
    velocity_vector    = segment.state.conditions.frames.inertial.velocity_vector
    orientation_vector = segment.state.conditions.frames.body.inertial_rotations
    
    alpha           = orientation_vector[:,1]
    beta            = orientation_vector[:,2]
    throttle        = segment.state.conditions.propulsion.throttle
    
    # ----------------------------------------------------------------------------
    # Perturb each state variable
    # ----------------------------------------------------------------------------

    # ----------------------------------------------------------------------------    
    # Alpha perturbation
    
    perturbed_segment = deepcopy(segment)    
    alpha_plus        = alpha*(1+h)
    perturbed_segment.state.conditions.frames.body.inertial_rotations[:,1] = alpha_plus
    
    iterate = perturbed_segment.process.iterate
    iterate.conditions(perturbed_segment) 
    
    # set segment derivatives based on perturbed segment
    dAlpha = np.atleast_2d(alpha_plus-alpha).T
    dCL    = perturbed_segment.state.conditions.aerodynamics.lift_coefficient - segment.state.conditions.aerodynamics.lift_coefficient
    dCD    = perturbed_segment.state.conditions.aerodynamics.drag_coefficient - segment.state.conditions.aerodynamics.drag_coefficient

    dCL_dAlpha = dCL/dAlpha
    dCD_dAlpha = dCD/dAlpha
    
    segment.state.conditions.aero_derivatives.dCL_dAlpha = dCL_dAlpha
    segment.state.conditions.aero_derivatives.dCD_dAlpha = dCD_dAlpha
    
    
    # ----------------------------------------------------------------------------    
    # Beta perturbation
    
    perturbed_segment = deepcopy(segment)    
    beta_plus         = beta+h
    perturbed_segment.state.conditions.frames.body.inertial_rotations[:,2] = beta_plus
    
    iterate = perturbed_segment.process.iterate
    iterate.conditions(perturbed_segment) 
    
    # set segment derivatives based on perturbed segment
    dBeta  = np.atleast_2d(beta_plus-beta).T
    dCL    = perturbed_segment.state.conditions.aerodynamics.lift_coefficient - segment.state.conditions.aerodynamics.lift_coefficient
    dCD    = perturbed_segment.state.conditions.aerodynamics.drag_coefficient - segment.state.conditions.aerodynamics.drag_coefficient

    dCL_dBeta = dCL/dBeta
    dCD_dBeta = dCD/dBeta
    
    segment.state.conditions.aero_derivatives.dCL_dBeta = dCL_dBeta
    segment.state.conditions.aero_derivatives.dCD_dBeta = dCD_dBeta
    
    # ----------------------------------------------------------------------------    
    # Velocity magnitude perturbation
    
    perturbed_segment = deepcopy(segment)    
    Vx_plus           = velocity_vector[:,0]*(1+h)
    perturbed_segment.state.conditions.frames.inertial.velocity_vector[:,0] = Vx_plus
    
    iterate = perturbed_segment.process.iterate
    iterate.conditions(perturbed_segment) 
    
    # set segment derivatives based on perturbed segment
    dV = np.atleast_2d(Vx_plus-velocity_vector[:,0]).T
    dCL    = perturbed_segment.state.conditions.aerodynamics.lift_coefficient - segment.state.conditions.aerodynamics.lift_coefficient
    dCD    = perturbed_segment.state.conditions.aerodynamics.drag_coefficient - segment.state.conditions.aerodynamics.drag_coefficient

    dCL_dV = dCL/dV
    dCD_dV = dCD/dV
    
    segment.state.conditions.aero_derivatives.dCL_dV = dCL_dV
    segment.state.conditions.aero_derivatives.dCD_dV = dCD_dV    
    
    # ----------------------------------------------------------------------------    
    # Throttle perturbation
    
    perturbed_segment = deepcopy(segment)    
    throttle_plus     = throttle*(1+h)
    perturbed_segment.state.conditions.propulsion.throttle = throttle_plus
    
    iterate = perturbed_segment.process.iterate
    iterate.conditions(perturbed_segment) 
    
    # set segment derivatives based on perturbed segment
    dThrottle = throttle_plus-throttle
    dCL       = perturbed_segment.state.conditions.aerodynamics.lift_coefficient - segment.state.conditions.aerodynamics.lift_coefficient
    dCD       = perturbed_segment.state.conditions.aerodynamics.drag_coefficient - segment.state.conditions.aerodynamics.drag_coefficient

    dCL_dThrottle = dCL/dThrottle
    dCD_dThrottle = dCD/dThrottle
    
    segment.state.conditions.aero_derivatives.dCL_dThrottle = dCL_dThrottle
    segment.state.conditions.aero_derivatives.dCD_dThrottle = dCD_dThrottle      
    
    return 