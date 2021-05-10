## @ingroupMethods-Noise-Fidelity_One-Engine
# ground_proximity_effect.py
# 
# Created:  Jul 2015, C. Ilario
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------        
#   Imports
# ---------------------------------------------------------------------- 

import numpy as np

# ----------------------------------------------------------------------        
#   Ground Proximity Effect
# ---------------------------------------------------------------------- 

## @ingroupMethods-Noise-Fidelity_One-Engine
def ground_proximity_effect (Velocity_mixed,sound_ambient,theta_m,engine_height,Diameter_mixed,frequency):
    """This function calculates the ground proximity effect, in decibels, and is used for full-scale 
    engine test stand.
        
    Assumptions:
        N/A

    Source:
        N/A

    Inputs:
        Velocity_mixed  [m/s]
        sound_ambient   [SPL]
        theta_m         [rad]
        engine_height   [m]
        Diameter_mixed  [m]
        frequency       [1/s]

    Outputs:
        GPROX_m         [dB]

    Properties Used:
        N/A 
    """ 
    # Ground proximity is applied only for the mixed jet component
    GPROX_m = (5*Velocity_mixed/sound_ambient)*np.exp(-(9*(theta_m/np.pi)-6.75)**2- \
        ((engine_height/Diameter_mixed)-2.5)**2)*(1+(np.sin((np.pi*engine_height*frequency/sound_ambient)-np.pi/2))**2)/ \
        (2+np.abs((engine_height*frequency/sound_ambient)-1))

    return GPROX_m