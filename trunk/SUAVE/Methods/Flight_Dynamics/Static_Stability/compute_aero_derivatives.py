## @ingroup Methods-Flight_Dynamics-Static_Stability
# compute_aero_derivatives.py
# 
# Created:   Aug 2021, R. Erhard
# Modified: 
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np 
from copy import deepcopy

## @ingroup Methods-Flight_Dynamics-Static_Stability
def compute_aero_derivatives(segment, h=1e-4): 
    """This function takes an computes the aerodynamic derivatives about a segment.
    
    Assumptions:
       Linearized equations are used for each state variable

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
    # check for surrogate
    surrogate_used = segment.analyses.aerodynamics.settings.use_surrogate
    
    # extract states from converged mission segment
    orientation_vector = segment.state.conditions.frames.body.inertial_rotations
    
    pitch     = orientation_vector[:,1]
    psi       = orientation_vector[:,2]      # heading 
    throttle  = segment.state.conditions.propulsion.throttle
    
    n_cpts    = len(pitch)
    
    # ----------------------------------------------------------------------------
    # Perturb each state variable
    # ----------------------------------------------------------------------------

    # ----------------------------------------------------------------------------    
    # Alpha perturbation
    
    perturbed_segment = deepcopy(segment)    
    pitch_plus        = pitch*(1+h)
    perturbed_segment.state.conditions.frames.body.inertial_rotations[:,1] = pitch_plus
    
    iterate = perturbed_segment.process.iterate
    iterate.conditions(perturbed_segment) 
    
    # set segment derivatives based on perturbed segment
    dAlpha = perturbed_segment.state.conditions.aerodynamics.angle_of_attack - segment.state.conditions.aerodynamics.angle_of_attack
    dCL    = perturbed_segment.state.conditions.aerodynamics.lift_coefficient - segment.state.conditions.aerodynamics.lift_coefficient
    
    if surrogate_used:
        dCM_dAlpha = segment.state.conditions.stability.static.Cm_alpha
    else:
        # use VLM outputs directly
        dCM    = perturbed_segment.state.conditions.aerodynamics.moment_coefficient - segment.state.conditions.aerodynamics.moment_coefficient
        dCM_dAlpha = dCM/dAlpha
        
    # propeller derivatives
    dCT, dCP = propeller_derivatives(segment, perturbed_segment, n_cpts)
        
    dCL_dAlpha = dCL/dAlpha
    dCT_dAlpha = dCT/dAlpha[None,:,:]
    dCP_dAlpha = dCP/dAlpha[None,:,:]
    
    segment.state.conditions.aero_derivatives.dCL_dAlpha = dCL_dAlpha
    segment.state.conditions.aero_derivatives.dCM_dAlpha = dCM_dAlpha
    segment.state.conditions.aero_derivatives.dCT_dAlpha = dCT_dAlpha
    segment.state.conditions.aero_derivatives.dCP_dAlpha = dCP_dAlpha
    
    
    # ----------------------------------------------------------------------------    
    # Beta perturbation
    
    perturbed_segment = deepcopy(segment)    
    psi_plus          = psi+h
    perturbed_segment.state.conditions.frames.body.inertial_rotations[:,2] = psi_plus
    
    iterate = perturbed_segment.process.iterate
    iterate.conditions(perturbed_segment) 
    
    # set segment derivatives based on perturbed segment
    dBeta  = perturbed_segment.state.conditions.aerodynamics.side_slip_angle - segment.state.conditions.aerodynamics.side_slip_angle
    
    # roll and yaw moment coefficient derivatives
    if surrogate_used:
        # check for roll and yaw moment coefficient derivatives in surrogate outputs
        if 'Cn_beta' in segment.state.conditions.stability.static.keys():
            dCn_dBeta = segment.state.conditions.stability.static.Cn_beta
        else:
            print("No dCn_dBeta in surrogate output. dCn_dBeta not included in aerodynamic derivatives.")
            dCn_dBeta = None
        if 'Cl_beta' in segment.state.conditions.stability.static.keys():
            dCl_dBeta = segment.state.conditions.stability.static.Cl_beta 
        else:
            print("No dCl_dBeta in surrogate output. dCl_dBeta not included in aerodynamic derivatives.")
            dCl_dBeta = None            
    else:
        # use VLM outputs directly
        dCn = perturbed_segment.state.conditions.stability.static.yawing_moment_coefficient - segment.state.conditions.stability.static.yawing_moment_coefficient
        dCl = perturbed_segment.state.conditions.stability.static.rolling_moment_coefficient - segment.state.conditions.stability.static.rolling_moment_coefficient
        dCn_dBeta = dCn/dBeta
        dCl_dBeta = dCl/dBeta
        
    # check for propellers
    dCT, dCP = propeller_derivatives(segment, perturbed_segment, n_cpts)  

    dCT_dBeta = dCT/dBeta
    dCP_dBeta = dCP/dBeta
    
    segment.state.conditions.aero_derivatives.dCn_dBeta = dCn_dBeta
    segment.state.conditions.aero_derivatives.dCl_dBeta = dCl_dBeta
    segment.state.conditions.aero_derivatives.dCT_dBeta = dCT_dBeta
    segment.state.conditions.aero_derivatives.dCP_dBeta = dCP_dBeta
    
    
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

    # check for propellers
    dCT, dCP = propeller_derivatives(segment, perturbed_segment, n_cpts)  

    dCL_dThrottle = dCL/dThrottle
    dCD_dThrottle = dCD/dThrottle
    dCT_dThrottle = dCT/dThrottle
    dCP_dThrottle = dCP/dThrottle
    
    segment.state.conditions.aero_derivatives.dCL_dThrottle = dCL_dThrottle
    segment.state.conditions.aero_derivatives.dCD_dThrottle = dCD_dThrottle      
    segment.state.conditions.aero_derivatives.dCT_dThrottle = dCT_dThrottle    
    segment.state.conditions.aero_derivatives.dCP_dThrottle = dCP_dThrottle         
    
    return 

def propeller_derivatives(segment, perturbed_segment, n_cpts):
    props = segment.state.conditions.noise.sources.propellers
    perturbed_props = perturbed_segment.state.conditions.noise.sources.propellers
    dCT = np.zeros((len(props),n_cpts,1))
    dCP = np.zeros((len(props),n_cpts,1))
    for i in range(len(props)):
        prop_key       = list(props.keys())[i]
        prop           = props[prop_key]
        perturbed_prop = perturbed_props[prop_key]
        
        dCT[i,:,:] = perturbed_prop.thrust_coefficient - prop.thrust_coefficient
        dCP[i,:,:] = perturbed_prop.power_coefficient - prop.power_coefficient
    
    return dCT, dCP