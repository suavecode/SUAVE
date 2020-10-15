## @ingroupMethods-Noise-Fidelity_One-Noise_Tools
# approach_noise.py
# 
# Created:  Oct 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np 
from SUAVE.Methods.Noise.Fidelity_One.Airframe.noise_airframe_Fink   import noise_airframe_Fink 
from SUAVE.Methods.Noise.Fidelity_One.Engine.noise_SAE               import noise_SAE  
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.noise_geometric    import noise_geometric 

# ----------------------------------------------------------------------        
#   NOISE CALCULATION
# ----------------------------------------------------------------------

## @ingroupMethods-Noise-Fidelity_One-Noise_Tools
def compute_noise(config,analyses,noise_segment):
    """This method calculates approach noise of a turbofan aircraft
            
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
    turbofan = config.propulsors['turbofan']
    
    engine_flag    = config.engine_flag  #remove engine noise component from the approach segment
    
    geometric      = noise_geometric(noise_segment,analyses,config)
    
    airframe_noise = noise_airframe_Fink(config,analyses,noise_segment)  

    engine_noise   = noise_SAE(turbofan,noise_segment,config,analyses)

    noise_sum      = 10. * np.log10(10**(airframe_noise[0]/10)+ (engine_flag)*10**(engine_noise[0]/10))

    return noise_sum
