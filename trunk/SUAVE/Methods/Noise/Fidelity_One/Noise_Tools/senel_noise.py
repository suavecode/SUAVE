# senel_noise.py
# 
# Created:  Jul 2015, C. Ilario

# ----------------------------------------------------------------------        
#   Imports
# ---------------------------------------------------------------------- 

import numpy as np

# ----------------------------------------------------------------------        
#   SENEL Noise Metric
# ---------------------------------------------------------------------- 

def senel_noise(SPLt_dBA_max):
    """This method calculates de effective perceived noise level (EPNL) based on a time history PNLT
     (Perceived Noise Level with Tone Correction).

        Inputs:
                    PNLT                     - Perceived Noise Level with Tone Correction

                Outputs: 
                    EPNL                     - Effective Perceived Noise Level in EPNdB"""
                    
                    
    #Maximum PNLT on the time history data    
    dBA_max = np.max(SPLt_dBA_max)
    
    #Calculates the number of discrete points on the trajectory
    nsteps   = len(SPLt_dBA_max)    
    
    #Exclude sources that are not being calculated or doesn't contribute for the total noise of the aircraft
    if all(SPLt_dBA_max==0):
        SENEL = 0
        return(SENEL)

    #Finding the time duration for the noise history where PNL is higher than the maximum PNLT - 10 dB
    i = 0
    while SPLt_dBA_max[i]<=(dBA_max-10) and i<=nsteps:
        i = i+1
    t1 = i #t1 is the first time interval
    i  = i+1

    #Correction for PNLTM-10 when it falls outside the limit of the data
    if SPLt_dBA_max[nsteps-1]>=(dBA_max-10):
        t2=nsteps-2
    else:
        while i<=nsteps and SPLt_dBA_max[i]>=(dBA_max-10):
            i = i+1
        t2 = i-1 #t2 is the last time interval
                
    #The time duration where the noise is higher than the maximum PNLT - 10 dB is:
    time_interval = (t2-t1)*0.5
    
    #Calculates the integral of the PNLT which between t1 and t2 points
    sumation = 0
    for i in range (t1-1,t2+1):
        sumation = 10**(SPLt_dBA_max[i]/10)+sumation
        
    SENEL = 10*np.log10(sumation)
    
    return (SENEL)    