## @ingroupMethods-Noise-Fidelity_One-Noise_Tools
# SPL_harmonic_to_third_octave.py
# 
# Created:  Jul 2015, C. Ilario
# Modified: Jan 2016, E. Botero

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
        SPL     - Sound Pressure Level             [dB] 

    Outputs: [dB] 

    Properties Used:
        N/A 
        
    """  
    cf = settings.center_frequencies
    lf = settings.lower_frequencies
    uf = settings.upper_frequencies
    
    dim_cpt          = len(SPL[:,0,0])
    dim_prop         = len(SPL[0,:,0])
    num_cf           = len(cf)
    num_f            = len(f)
    SPL_third_octave = np.zeros((dim_cpt,dim_prop,num_cf)) 
    
    # to vectorize 
    for i in range(dim_cpt):
        for j in range(dim_prop):
            for k in range(num_cf):
                SPL_in_range = []
                for l in range(num_f):  
                    if ((lf[k] <= f[l]) and (f[l] <= uf[k])) and (SPL[i,j,l] != 0) :   
                        SPL_in_range.append(SPL[i,j,l]) 
                    if len(SPL_in_range) > 0:
                        SPL_third_octave[i,j,k] = SPL_arithmetic(np.atleast_2d(np.array(SPL_in_range))) 
                    
    return SPL_third_octave
