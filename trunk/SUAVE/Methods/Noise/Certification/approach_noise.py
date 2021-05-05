## @ingroup Methods-Noise-Certification
# approach_noise.py
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

## @ingroup Methods-Noise-Certification 
def approach_noise(analyses,noise_configs):  
    """This method calculates approach noise of a turbofan aircraft
            
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
    mission                                                       = analyses.missions.landing
    approach_initialization                                       = mission.evaluate()   
    n_points                                                      = np.ceil(approach_initialization.segments.descent.conditions.frames.inertial.time[-1] /0.5 +1)
    mission.npoints_takeoff_sign                                  = np.sign(n_points) 
    mission.segments.descent.state.numerics.number_control_points = int(np.minimum(200, np.abs(n_points))[0])

    # Set up analysis 
    noise_segment                                  = mission.segments.descent 
    noise_segment.analyses.noise.settings.approach = True
    noise_analyses                                 = noise_segment.analyses 
    noise_settings                                 = noise_segment.analyses.noise.settings
    noise_config                                   = noise_configs.landing  
    noise_config.engine_flag                       = True
    noise_config.print_output                      = 0
    noise_config.output_file                       = 'Noise_Approach.dat'
    noise_config.output_file_engine                = 'Noise_Approach_Engine.dat'    

    noise_result_approach = compute_noise(noise_config,noise_analyses,noise_segment,noise_settings)   
        
    return noise_result_approach
