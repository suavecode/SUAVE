## @ingroupMethods-Noise-Fidelity_One-Noise_Tools
# approach_noise.py
# 
# Created:  Oct 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np 
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.noise_geometric    import noise_geometric 
from SUAVE.Methods.Noise.Fidelity_One.Engine                         import noise_SAE  
from SUAVE.Methods.Noise.Fidelity_One.Airframe.noise_airframe_Fink   import noise_airframe_Fink 

# ----------------------------------------------------------------------        
#   NOISE CALCULATION
# ----------------------------------------------------------------------

## @ingroupMethods-Noise-Fidelity_One-Noise_Tools
def compute_noise(config,noise_segment,noise_analyses):
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
    
    
    engine_flag    = config.engine_flag  #remove engine noise component from the approach segment 
    turbofan       = config.propulsors['turbofan']
    engine_noise   = noise_SAE(turbofan,noise_segment,noise_analyses,config)
    
    airframe_noise = noise_airframe_Fink(noise_segment,noise_analyses,config ) 

    noise_sum      = 10. * np.log10(10**(airframe_noise.EPNL_total/10)+ (engine_flag)*10**(engine_noise.EPNL_total/10))

    return noise_sum
