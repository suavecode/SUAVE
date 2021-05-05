## @ingroupMethods-Noise-Fidelity_One-Noise_Tools
# dbA_noise.py
# 
# Created:  Jul 2015, C. Ilario
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  dbA Noise
# ----------------------------------------------------------------------

## @ingroupMethods-Noise-Fidelity_One-Noise_Tools
def dbA_noise(SPL):
    """This method calculates the A-weighted level from a 1/3 octave band noise spectra 
            
    Assumptions:
        N/A

    Source:
        N/A 

    Inputs:
        SPL     - Sound Pressure Level in 1/3 octave band

    Outputs: [dB]
        SPL_dbA - A-weighted Sound Pressure Level in dBA 

    Properties Used:
        N/A 
        
    """ 
    # Matrix with the dbA attenuation factor for each octave band frequnecy ranging from 50Hz to 10000Hz
    dbA_attenuation = np.array((-30.2,-26.2,-22.5,-19.1,-16.1,-13.4,-10.9,-8.6,-6.6,-4.8,-3.2,-1.9,-0.8,0,0.6,1,1.2,1.3,1.2,1,0.5,-0.1,-1.1,-2.5))
    
    #Calculation the SPL_dbA
    SPL_dbA = SPL+dbA_attenuation
        
    return SPL_dbA

def A_weighting(SPL,f): 
    """This method calculates the A-weighted SPL given its stectra
    
    Assumptions:
        N/A

    Source:
        IEC 61672-1:2013 Electroacoustics - Sound level meters - Part 1: Specifications. IEC. 2013.

    Inputs:
        SPL     - Sound Pressure Level             [dB] 

    Outputs: [dB]
        SPL_dbA - A-weighted Sound Pressure Level  [dBA] 

    Properties Used:
        N/A 
        
    """    
    Ra_f       = ((12194**2)*(f**4))/ (((f**2)+(20.6**2)) * ((f**2)+(12194**2)) * (((f**2) + 107.7**2)**0.5)*(((f**2)+ 737.9**2)**0.5)) 
    A_f        =  2.0  + 20*np.log10(Ra_f) 
    SPL_dBA = SPL + A_f
    return SPL_dBA
