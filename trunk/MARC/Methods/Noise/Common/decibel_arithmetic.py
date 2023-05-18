## @ingroup Methods-Noise-Fidelity_Zero-Common
# decibel_arithmetic.py
# 
# Created: Apr 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np 
# -----------------------------------------------------------------------
# Decibel Arithmetic
# -----------------------------------------------------------------------
## @ingroup Methods-Noise-Fidelity_Zero-Common
def pressure_ratio_to_SPL_arithmetic(p_pref_total):
    ''' This computes the total SPL given mutiple acoustic pressure ratios 
    of one of mutiple sources
    
    Assumptions:
        None

    Source:
        None

    Inputs:
        Pressure Ratios       [unitless]

    Outputs: 
        Sound Pressure Level  [decibel]

    Properties Used:
        N/A 
    
    '''
    SPL_total = 10*np.log10(np.nansum(p_pref_total, axis = 3))
    return SPL_total

## @ingroup Methods-Noise-Fidelity_Zero-Common
def SPL_arithmetic(SPL, sum_axis = 2):
    '''This computes the total SPL from multiple sources 
    using decibel arithmetic  
    
    Assumptions:
        None

    Source:
        None

    Inputs:
        SPL  -  Sound Pressure Level        [dB]

    Outputs: 
        SPL  -  Sound Pressure Level        [dB]
    
    Properties Used:
        N/A 
    
    '''
    if SPL.ndim == 1:
        SPL_total = SPL 
    else:
        p_prefs   = 10**(SPL/10)
        SPL_total = 10*np.log10(np.nansum(p_prefs, axis = sum_axis))
        
    return SPL_total