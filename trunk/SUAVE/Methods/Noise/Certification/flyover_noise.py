## @ingroup Methods-Noise-Certification
# flyover_noise.py
# 
# Created:  Oct 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np 
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.compute_noise import compute_noise

# ----------------------------------------------------------------------        
#   Flyover noise
# ----------------------------------------------------------------------     

## @ingroup Methods-Noise-Certification 
def flyover_noise(analyses,noise_configs):  
    """This method calculates flyover noise of a turbofan aircraft
            
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
    mission                                                     = analyses.missions.takeoff 
    takeoff_initialization                                      = mission.evaluate() 
    n_points                                                    = np.ceil(takeoff_initialization.segments.climb.conditions.frames.inertial.time[-1] /0.5 +1)
    mission.npoints_takeoff_sign                                = np.sign(n_points) 
    mission.segments.climb.state.numerics.number_control_points = int(np.minimum(200, np.abs(n_points))[0])

    # Set up analysis 
    noise_segment                                  = mission.segments.climb  
    noise_segment.analyses.noise.settings.sideline = False  
    noise_segment.analyses.noise.settings.flyover  = True 
    noise_settings                                 = noise_segment.analyses.noise.settings 
    noise_config                                   = noise_configs.takeoff
    noise_analyses                                 = noise_segment.analyses 
    noise_config.engine_flag                       = True
    noise_config.print_output                      = 0
    noise_config.output_file                       = 'Noise_Flyover_climb.dat'
    noise_config.output_file_engine                = 'Noise_Flyover_climb_Engine.dat' 
    noise_config.engine_flag                       = True
    
    if mission.npoints_takeoff_sign == -1:
        noise_result_takeoff_FL_clb = 500. + noise_segment.missions.sideline_takeoff.segments.climb.state.numerics.number_control_points
    else:    
        noise_result_takeoff_FL_clb = compute_noise(noise_config,noise_analyses,noise_segment,noise_settings)   

    noise_segment                   = mission.segments.cutback 
    noise_config                    = noise_configs.cutback
    noise_config.print_output       = 0
    noise_config.engine_flag        = True
    noise_config.output_file        = 'Noise_Flyover_cutback.dat'
    noise_config.output_file_engine = 'Noise_Flyover_cutback_Engine.dat'
    
    if mission.npoints_takeoff_sign == -1:
        noise_result_takeoff_FL_cutback = 500. + noise_segment.missions.sideline_takeoff.segments.climb.state.numerics.number_control_points
    else:                                       
        noise_result_takeoff_FL_cutback = compute_noise(noise_config,noise_analyses,noise_segment,noise_settings)   
 
    noise_result_takeoff_FL = 10. * np.log10(10**(noise_result_takeoff_FL_clb/10)+10**(noise_result_takeoff_FL_cutback/10))

    return noise_result_takeoff_FL
