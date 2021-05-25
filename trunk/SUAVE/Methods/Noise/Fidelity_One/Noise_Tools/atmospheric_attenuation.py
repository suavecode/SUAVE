## @ingroupMethods-Noise-Fidelity_One-Noise_Tools
# atmospheric_attenuation.py
# 
# Created:  Jul 2015, C. Ilario
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Atmospheric Attenuation
# ----------------------------------------------------------------------

## @ingroupMethods-Noise-Fidelity_One-Noise_Tools
def atmospheric_attenuation(dist):
    """ This calculates a the atmospheric attenuation of sound for each frequency band
    as a function of the distance from the source to the observer. 
    
    Assumptions:
       N/A

    Source:
       SAE Model for a standard day.
        
    Inputs:
        dist - Array with the distance vector from the aircraft (source) to the 
            microphone position (observer)                                              [m]

    Outputs: 
        delta_spl - The delta sound pressure level to be reduced from the lossless SPL 
        condition                                                                       [dB]
    
    Properties Used:
        None 
    """
    
    # Atmospheric attenuation factor for a 70% humidity and 25 Celsius at 1000ft - Based SAE model
    Att_dB = np.array((0.09,0.11,0.14,0.17,0.22,0.28,0.35,0.44,0.55,0.7,0.88,1.11,1.42,1.78,2.24,2.88,3.64,4.6,5.89,7.63,8.68,11.08,14.87,20.61))
    
    # Calculates de delta SPL as a function of the distance
    delta_spl = Att_dB*(dist)/1000
    
    return delta_spl
