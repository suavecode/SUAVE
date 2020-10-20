## @ingroupMethods-Noise-Certification
# sideline_noise.py
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
def sideline_noise(mission,noise_config):  
    """This method calculates sideline noise of a turbofan aircraft
            
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
    sideline_initialization_results                             = mission.evaluate() 
    n_points                                                    = np.ceil(sideline_initialization_results.segments.climb.conditions.frames.inertial.time[-1] /0.5 +1) 
    mission.npoints_sideline_sign                               = np.sign(n_points) 
    mission.segments.climb.state.numerics.number_control_points = np.minimum(200, np.abs(n_points))  

    # Set up analysis 
    noise_segment                                  = mission.segments.climb  
    noise_analyses                                 = noise_segment.analyses 
    noise_segment.analyses.noise.settings.sideline = True
    noise_segment.analyses.noise.settings.flyover  = False 

    # Determine the x0
    x0              = 0.    
    position_vector = noise_segment.conditions.frames.inertial.position_vector
    degree          = 3
    coefs           = np.polyfit(-position_vector[:,2],position_vector[:,0],degree)
    for idx,coef in enumerate(coefs):
        x0 += coef * 304.8 ** (degree-idx) 

    noise_segment.analyses.noise.settings.mic_x_position = x0   
    noise_config.engine_flag                             = True 

    if mission.npoints_sideline_sign == -1:
        noise_result_takeoff_SL = 500. + noise_segment.missions.sideline_takeoff.segments.climb.state.numerics.number_control_points
    else:
        noise_result_takeoff_SL = compute_noise(noise_config,noise_segment,noise_analyses)    

    return noise_result_takeoff_SL
