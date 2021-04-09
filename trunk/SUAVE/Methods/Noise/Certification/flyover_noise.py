## @ingroupMethods-Noise-Certification
# flyover_noise.py
# 
# Created:  Oct 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np 
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.compute_noise import compute_noise

# ----------------------------------------------------------------------        
#   Approach noise
# ----------------------------------------------------------------------     

## @ingroupMethods-Noise-Certification 
def flyover_noise(mission,noise_config):  
    """This method calculates flyover noise of a turbofan aircraft
            
    Assumptions:
        N/A

    Source:
        N/A 

    Inputs:
        mission
        aircraft configuration 

    Outputs: 
        SPL    -  [dB]

    Properties Used:
        N/A 
        
    """ 
    # Update number of control points for noise 
    takeoff_initialization                                      = mission.evaluate() 
    n_points                                                    = np.ceil(takeoff_initialization.segments.climb.conditions.frames.inertial.time[-1] /0.5 +1)
    mission.npoints_takeoff_sign                                = np.sign(n_points) 
    mission.segments.climb.state.numerics.number_control_points = np.minimum(200, np.abs(n_points))

    # Set up analysis 
    noise_segment                                  = mission.segments.climb 
    noise_analyses                                 = noise_segment.analyses 
    noise_segment.analyses.noise.settings.sideline = False  
    noise_segment.analyses.noise.settings.flyover  = True
    
    noise_config.engine_flag = True

    if mission.npoints_takeoff_sign == -1:
        noise_result_takeoff_FL_clb = 500. + noise_segment.missions.sideline_takeoff.segments.climb.state.numerics.number_control_points
    else:    
        noise_result_takeoff_FL_clb = compute_noise(noise_config,noise_segment,noise_analyses)   


    if mission.npoints_takeoff_sign == -1:
        noise_result_takeoff_FL_cutback = 500. + noise_segment.missions.sideline_takeoff.segments.climb.state.numerics.number_control_points
    else:                                       
        noise_result_takeoff_FL_cutback = compute_noise(noise_config,noise_segment,noise_analyses)   

    noise_result_takeoff_FL = 10. * np.log10(10**(noise_result_takeoff_FL_clb/10)+10**(noise_result_takeoff_FL_cutback/10))

    return noise_result_takeoff_FL
