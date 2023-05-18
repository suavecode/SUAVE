## @ingroup Methods-Noise-Metrics
# A_weighting_metric.py
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

## @ingroup Methods-Noise-Metrics
def A_weighting_metric(SPL,f): 
    """This method calculates the A-weighted SPL given its stectra
    
    Assumptions:
        N/A

    Source:
        IEC 61672-1:2013 Electroacoustics - Sound level meters - Part 1: Specifications. IEC. 2013.

    Inputs:
        SPL     - Sound Pressure Level             [dB] 

    Outputs: [dB]
        SPL_dBA - A-weighted Sound Pressure Level  [dBA] 

    Properties Used:
        N/A 
        
    """    
    Ra_f       = ((12194**2)*(f**4))/ (((f**2)+(20.6**2)) * ((f**2)+(12194**2)) * (((f**2) + 107.7**2)**0.5)*(((f**2)+ 737.9**2)**0.5)) 
    A_f        =  2.0  + 20*np.log10(Ra_f) 
    SPL_dBA    = SPL + A_f
    return SPL_dBA
