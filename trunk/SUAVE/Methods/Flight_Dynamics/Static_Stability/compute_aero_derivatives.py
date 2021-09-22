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
def compute_aero_derivatives(segment): 
    """This function computes the aerodynamic derivatives of the aircraft about a 
    mission segment, and stores them in the aero_derivatives data structure associated
    with the state conditions of the given mission segment. All derivatives are 
    computed using forward difference.
    
    Assumptions:
       Linearized equations are used for each state variable

    Source:
      N/A

    Inputs:
      segment                SUAVE mission segment
      
    Outputs: 
       segment.state.conditions.aero_derivatives
         .dCL_dAlpha       -   lift-curve slope                                                            [-]  
         .dCM_dAlpha       -   derivative of pitching moment coefficient with respect to angle of attack   [-] 
         .dCT_dAlpha       -   derivative of rotor thrust coefficient with respect to angle of attack      [-] 
         .dCP_dAlpha       -   derivative of rotor power coefficient with respect to angle of attack       [-] 
         .dCn_dBeta        -   derivative of yawing moment coefficient with respect to sideslip angle      [-] 
         .dCl_dBeta        -   derivative of pitching moment with respect to angle of attack               [-] 
         .dCT_dBeta        -   derivative of rotor thrust coefficient with respect to sideslip angle       [-] 
         .dCP_dBeta        -   derivative of rotor power coefficient with respect to sideslip angle        [-] 
         .dCL_dThrottle    -   derivative of lift coefficient with respect to throttle                     [-] 
         .dCD_dThrottle    -   derivative of drag coefficient with respect to throttle                     [-] 
         .dCT_dThrottle    -   derivative of rotor thrust coefficient with respect to throttle             [-] 
         .dCP_dThrottle    -   derivative of rotor power coefficient with respect to throttle              [-] 
           

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
    
    # find main wing flap deflection
    delta     = segment.analyses.aerodynamics.geometry.wings.main_wing.control_surfaces.flap.deflection
    
    n_cpts    = len(pitch)
    
    # ----------------------------------------------------------------------------
    # Perturb each state variable
    # ----------------------------------------------------------------------------
    h = 1e-4
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
    

    # ----------------------------------------------------------------------------    
    # Flap deflection perturbation
    
    perturbed_segment = deepcopy(segment)    
    delta_plus        = delta + 0.1
    perturbed_segment.analyses.aerodynamics.geometry.wings.main_wing.control_surfaces.flap.deflection = delta_plus
    
    iterate = perturbed_segment.process.iterate
    iterate.conditions(perturbed_segment) 
    
    # set segment derivatives based on perturbed segment
    dDelta_Flap     = perturbed_segment.analyses.aerodynamics.geometry.wings.main_wing.control_surfaces.flap.deflection - segment.analyses.aerodynamics.geometry.wings.main_wing.control_surfaces.flap.deflection
    dCL             = perturbed_segment.state.conditions.aerodynamics.lift_coefficient - segment.state.conditions.aerodynamics.lift_coefficient
    dCL_dDelta_Flap = dCL/dDelta_Flap
    
    segment.state.conditions.aero_derivatives.dCL_dDelta_Flap = dCL_dDelta_Flap
    
    # ----------------------------------------------------------------------------    
    # Slat deflection perturbation
    
    perturbed_segment = deepcopy(segment)    
    delta_plus        = delta + 0.1
    perturbed_segment.analyses.aerodynamics.geometry.wings.main_wing.control_surfaces.slat.deflection = delta_plus
    
    iterate = perturbed_segment.process.iterate
    iterate.conditions(perturbed_segment) 
    
    # set segment derivatives based on perturbed segment
    dDelta_Slat     = perturbed_segment.analyses.aerodynamics.geometry.wings.main_wing.control_surfaces.flap.deflection - segment.analyses.aerodynamics.geometry.wings.main_wing.control_surfaces.flap.deflection
    dCL             = perturbed_segment.state.conditions.aerodynamics.lift_coefficient - segment.state.conditions.aerodynamics.lift_coefficient
    dCL_dDelta_Slat = dCL/dDelta_Slat
    
    segment.state.conditions.aero_derivatives.dCL_dDelta_Slat = dCL_dDelta_Slat    
    
    # ----------------------------------------------------------------------------    
    # Velocity magnitude perturbation
    vinf                = segment.state.conditions.frames.inertial.velocity_vector
    vmag                = np.linalg.norm(vinf,axis=1)
    psi                 = np.arctan2(vinf[:,2],vinf[:,0])
    
    perturbed_segment   = deepcopy(segment)    
    vmag_plus           = vmag*(1+h) 
    perturbed_segment.state.conditions.frames.inertial.velocity_vector[:,0] = vmag_plus*np.cos(psi)
    perturbed_segment.state.conditions.frames.inertial.velocity_vector[:,2] = vmag_plus*np.sin(psi)
        
    iterate = perturbed_segment.process.iterate
    iterate.conditions(perturbed_segment)
    
    # set segment derivatives based on perturbed segment
    dV     = perturbed_segment.state.conditions.freestream.velocity - segment.state.conditions.frames.inertial.velocity_vector
    dCL    = perturbed_segment.state.conditions.aerodynamics.lift_coefficient - segment.state.conditions.aerodynamics.lift_coefficient
    dCL_dV = dCL/dV   

    segment.state.conditions.aero_derivatives.dCL_dV = dCL_dV      
    
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