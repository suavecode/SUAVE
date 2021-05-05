## @ingroup Methods-Noise-Certification
# sideline_noise.py
# 
# Created:  Oct 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np 
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.compute_noise import compute_noise

# ----------------------------------------------------------------------        
#   Sideline noise
# ----------------------------------------------------------------------     

## @ingroup Methods-Noise-Certification 
def sideline_noise(analyses,noise_configs):  
    """This method calculates sideline noise of a turbofan aircraft
            
    Assumptions:
        N/A

    Source:
        N/A 

    Inputs:
        analyses        - data structure of SUAVE analyses                [None]
        noise_configs   - data structure for SUAVE vehicle configurations [None]

    Outputs: 
        SPL             - sound pressure level                            [dB]

    Properties Used:
        N/A 
        
    """ 

    # Update number of control points for noise 
    mission                                                     = analyses.missions.sideline_takeoff 
    sideline_initialization_results                             = mission.evaluate() 
    n_points                                                    = np.ceil(sideline_initialization_results.segments.climb.conditions.frames.inertial.time[-1] /0.5 +1) 
    mission.npoints_sideline_sign                               = np.sign(n_points) 
    mission.segments.climb.state.numerics.number_control_points = int(np.minimum(200, np.abs(n_points))[0])

    # Set up analysis 
    noise_segment                                  = mission.segments.climb  
    noise_segment.analyses.noise.settings.sideline = True
    noise_segment.analyses.noise.settings.flyover  = False  
    noise_analyses                                 = noise_segment.analyses 
    noise_settings                                 = noise_segment.analyses.noise.settings 
    noise_config                                   = noise_configs.takeoff 
    noise_config.engine_flag                       = True
    noise_config.print_output                      = 0
    noise_config.output_file                       = 'Noise_Sideline.dat'
    noise_config.output_file_engine                = 'Noise_Sideline_Engine.dat'
    
    
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
        noise_result_takeoff_SL = compute_noise(noise_config,noise_analyses,noise_segment,noise_settings)    

    return noise_result_takeoff_SL
