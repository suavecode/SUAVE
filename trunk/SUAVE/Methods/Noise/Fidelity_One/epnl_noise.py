#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:     
#
# Author:      CARIDSIL
#
# Created:     21/07/2015
# Copyright:   (c) CARIDSIL 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np

def epnl_noise(PNLT):
    
    PNLT_max = np.max(PNLT)
    
    nsteps = len(PNLT)    
    
    #Exclude sources that are not being calculated or doesn't contribute for the total noise of the aircraft
    if all(PNLT==0):
        EPNL=0
        return(EPNL)

    #Finding the time duration for the noise history where PNL is higher than the maximum PNLT - 10 dB
    i=0
    while PNLT[i]<=(PNLT_max-10) and i<=nsteps:
        i=i+1
    t1=i #t1 is the first time interval
    i=i+1

    #Correction for PNLTM-10 when it falls outside the limit of the data
    if PNLT[nsteps-1]>=(PNLT_max-10):
        t2=nsteps-2
    else:
        while i<=nsteps and PNLT[i]>=(PNLT_max-10):
              i=i+1
        t2=i-1 #t2 is the last time interval
                
    #The time duration where the noise is higher than the maximum PNLT - 10 dB is:
    time_interval=(t2-t1)*0.5
    
    #Modification 31/07
    #duration_correction = 10*np.log10(np.sum(10**(PNLT/10)))-PNLT_max-13
   # EPNL=PNLT_max+duration_correction
    
    sumation=0
    for i in range (t1-1,t2+1):
        sumation=10**(PNLT[i]/10)+sumation
   
    duration_correction=10*np.log10(sumation)-PNLT_max-13
                
    
    EPNL=PNLT_max+duration_correction
    
    
    return (EPNL)    
    
#Test
#PNLT_test=np.array((66.96,68.34,69.73,71.11,72.5,73.92,75.34,76.81,78.96,81.35,83.84,86.75,89.8,92.4,93.75,93.09,90.49,87.37,84.8,82.92,81.43,80.09,78.89,77.8,76.71,75.61,74.75,74,73.27,72.54,71.79,71.08,70.36,69.62,68.9,68.19,67.47))
#test=epnl_noise(PNLT_test)

