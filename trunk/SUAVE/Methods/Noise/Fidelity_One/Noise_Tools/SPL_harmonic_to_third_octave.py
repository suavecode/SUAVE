## @ingroupMethods-Noise-Fidelity_One-Noise_Tools
# SPL_harmonic_to_third_octave.py
# 
# Created:  Jul 2015, C. Ilario
# Modified: Jan 2016, E. Botero
# Modified: Apr 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import SPL_arithmetic
# ----------------------------------------------------------------------
#  dbA Noise
# ----------------------------------------------------------------------

## @ingroupMethods-Noise-Fidelity_One-Noise_Tools 
def SPL_harmonic_to_third_octave(SPL,f,settings): 
    """This method converts the SPL spectrum from blade harmonic passing frequency
    to thrid octave spectrum
    
    Assumptions:
        N/A

    Source: 

    Inputs:
        SPL                    - sound pressure level                          [dB] 
        f                      - blade passing spectrum frequencies            [Hz]
        settings.    
            center_frequencies - center frequencies of the 1/3 octave spectrum [dB]  
            lower_frequencies  - lower frequencies of the 1/3 octave spectrum  [dB]
            upper_frequencies  - upper frequencies of the 1/3 octave spectrum  [dB]
        

    Outputs:
        SPL_third_octave       - SPL in the 1/3 octave spectrum                [dB] 

    Properties Used:
        N/A 
        
    """  
    # unpack 
    cf               = settings.center_frequencies
    lf               = settings.lower_frequencies
    uf               = settings.upper_frequencies
    
    dim_cpt          = len(SPL[:,0,0,0])
    dim_mic          = len(SPL[0,:,0,0])
    dim_prop         = len(SPL[0,0,:,0])
    num_cf           = len(cf)
    num_f            = len(f[0,:])
    SPL_third_octave = np.zeros((dim_cpt,dim_mic,dim_prop,num_cf)) 
    
    # loop through 1/3 octave spectra and sum up components 
    for i in range(dim_cpt):  
        for j in range(num_cf):
            SPL_in_range = np.empty(shape=[dim_mic,dim_prop,0 ])
            for k in range(num_f):  
                if (lf[j] <= f[i,k]) and (f[i,k] <= uf[j]):   
                    SPL_in_range = np.concatenate((SPL_in_range,np.atleast_3d(SPL[i,:,:,k])), axis = 2) 
                if len(SPL_in_range[0,0,:]) > 0:  
                    SPL_in_range[SPL_in_range != 0]  
                    SPL_third_octave[i,:,:,j] = SPL_arithmetic(SPL_in_range)   
                    
    return SPL_third_octave
