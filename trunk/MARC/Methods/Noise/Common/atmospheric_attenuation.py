## @ingroup Methods-Noise-Common
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

## @ingroup Methods-Noise-Common
def atmospheric_attenuation(dist,center_frequencies):
    """ This calculates a the atmospheric attenuation of sound for each frequency band
    as a function of the distance from the source to the observer using Average Atmospheric 
    Attenuation Rates from SAE-AIR-1845
    
    Assumptions:
       N/A

    Source:
       AEDT Technical Manual Table 11-1 (SAE-AIR-1845) 
        
    Inputs:
        dist - Array with the distance vector from the aircraft (source) to the 
            microphone position (observer)                                              [m]
        center_frequency - center frequencies of one-third octave band                  [Hz]

    Outputs: 
        delta_spl - The delta sound pressure level to be reduced from the lossless SPL 
        condition                                                                       [dB]
    
    Properties Used:
        None 
    """ 
    ctrl_pts  = len(dist)
    # Atmospheric attenuation factor for a 70% humidity and 25 Celsius at 1000ft - Based SAE model 
    Att_dB = np.array([0.033,0.033,0.033,0.066,0.066,0.098,0.131,0.131,
                       0.197,0.230,0.295,0.361,0.459,0.590,0.754,0.983,1.311,1.705,2.295,3.115,
                       3.607,5.246,7.213,9.836])
    if len(center_frequencies)>24: 
        no_Att_dB = np.zeros(len(center_frequencies)-24)
        Att_dB    = np.hstack((no_Att_dB,Att_dB)) 
    
    # Calculates de delta SPL as a function of the distance
    delta_spl = np.tile(Att_dB[None,:],(ctrl_pts,1))*(np.tile(dist[:,None],(1,len(Att_dB))))/100
    
    return delta_spl

