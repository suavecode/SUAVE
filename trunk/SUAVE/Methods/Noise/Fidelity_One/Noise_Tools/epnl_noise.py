## @ingroupMethods-Noise-Fidelity_One-Noise_Tools
# epnl_noise.py
# 
# Created:  Jul 2015, C. Ilario
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------        
#   Imports
# ---------------------------------------------------------------------- 

import numpy as np

# ----------------------------------------------------------------------        
#   EPNL Noise
# ---------------------------------------------------------------------- 

## @ingroupMethods-Noise-Fidelity_One-Noise_Tools
def epnl_noise(PNLT):
    """This method calculates de effective perceived noise level (EPNL) based on a
    time history Perceived Noise Level with Tone Correction (PNLT).
     
    Assumptions:
        N/A

    Source:
        N/A

    Inputs:
        PNLT - Perceived Noise Level with Tone Correction  [PNLdB]
     
     Outputs:
        EPNL - Effective Perceived Noise Level in          [EPNdB]
     
    Properties Used:
        N/A  
    """           
    # Maximum PNLT on the time history data    
    PNLT_max = np.max(PNLT)
    
    # Calculates the number of discrete points on the trajectory
    nsteps   = len(PNLT)    
    
    # Exclude sources that are not being calculated or doesn't contribute for the total noise of the aircraft
    if all(PNLT==0):
        EPNL = 0
        return(EPNL)

    # Finding the time duration for the noise history where PNL is higher than the maximum PNLT - 10 dB
    i = 0
    while PNLT[i]<=(PNLT_max-10) and i<=nsteps:
        i = i+1
    t1 = i #t1 is the first time interval
    i  = i+1

    # Correction for PNLTM-10 when it falls outside the limit of the data
    if PNLT[nsteps-1]>=(PNLT_max-10):
        t2=nsteps-2
    else:
        while i<=nsteps and PNLT[i]>=(PNLT_max-10):
            i = i+1
        t2 = i-1 #t2 is the last time interval 
    
    # Calculates the integral of the PNLT which between t1 and t2 points
    sumation = 0
    for i in range (t1-1,t2+1):
        sumation = 10**(PNLT[i]/10)+sumation
        
    # Duration Correction calculation
    duration_correction = 10*np.log10(sumation)-PNLT_max-13
                
    # Final EPNL calculation
    EPNL = PNLT_max+duration_correction
    
    return EPNL   