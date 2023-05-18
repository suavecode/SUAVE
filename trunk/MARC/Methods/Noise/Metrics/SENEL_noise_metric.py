## @ingroup Methods-Noise-Fidelity_Zero-Common
# SENEL_noise_metric.py
#
# Created:  Jul 2015, C. Ilario
# Created:  May 2023, M Clarke

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------  
from MARC.Core import Units
import numpy as np
from MARC.Methods.Noise.Common.decibel_arithmetic   import SPL_arithmetic
from MARC.Methods.Noise.Common.background_noise     import background_noise

# ----------------------------------------------------------------------        
#   SENEL Noise Metric
# ---------------------------------------------------------------------- 

## @ingroup Methods-Noise-Metrics
def SENEL_noise_metric(noise_data, flight_times = ['12:00:00'],time_period = 24*Units.hours):
    """This method calculates the Single Event Noise Exposure Level at all points in the computational domain

    Assumptions:
        None

    Source:
        None  
    
    Inputs: 
        SPL      - Noise level 
        
    Outputs: 
        SENEL    - Single Event Noise Exposure Level            [SENEL]
        
    Properties Used:
        N/A     
    """       
    

    if not hasattr(noise_data,'temporal_noise_exposure'): 
        SPL        = np.zeros_like(noise_data.SPL_dBA)
        SPL[:,:,:] = noise_data.SPL_dBA       
        t          = noise_data.time  
        N_gm_y     = noise_data.ground_microphone_y_resolution   
        N_gm_x     = noise_data.ground_microphone_x_resolution    
        time_step  = t[1]-t[0]
        
        # Compute Day-Night Sound Level and Noise Equivalent Noise   
        number_of_flights       = len(flight_times) 
        T                       = 15*Units.hours
        number_of_timesteps     = int(T/time_step) 
     
        TNE = np.zeros((number_of_timesteps,N_gm_x,N_gm_y))*np.nan   # cumulative noise exposure
        SPL[SPL == background_noise()]   =  np.nan
        for i in range(number_of_flights): 
            # get start time of flight
            t0  = int((np.float(flight_times[i].split(':')[0])*60*60 + \
                      np.float(flight_times[i].split(':')[1])*60 + \
                      np.float(flight_times[i].split(':')[2]) - 6*Units.hours)/time_step)    
            p_prefs_A               =  10**(TNE[t0:t0+len(t)][:,:,:,None]/10)
            p_prefs_B               =  10**(SPL[:,:,:,None]/10)
            C                       =  np.concatenate((p_prefs_A,p_prefs_B),axis = 3)
            TNE[t0:t0+len(t)]       =  10*np.log10(np.nansum(C,axis=3))  
            TNE[t0:t0+len(t)]       = SPL_arithmetic(np.concatenate((TNE[t0:t0+len(t)][:,:,:,None] , SPL[:,:,:,None]), axis=3), sum_axis=3) 
            
    
    else:
        time_step           = noise_data.time_step            
        TNE                 = noise_data.temporal_noise_exposure 
        number_of_timesteps = noise_data.number_of_timesteps
        timestamps          = noise_data.time_stamps
        N_gm_x              = len(TNE[0,:,0])
        N_gm_y              = len(TNE[0,0,:])
    
    # get matrix of maximum noise levels 
    SPL_max = np.max(TNE,axis = 0)
    
    # subtract 10 db to get bounds 
    SPL_max_min10 = SPL_max - 10
    
    # mask all noise values that are lower than L-10 level
    SPL_valid  = np.ma.masked_array(TNE, TNE >SPL_max_min10)
    SENEL      = SPL_arithmetic(SPL_valid,sum_axis=0)   
    
    # sum the noise 
    noise_data.SENEL                   = SENEL
    noise_data.temporal_noise_exposure = TNE 
    noise_data.time_step               = time_step
    noise_data.number_of_timesteps     = number_of_timesteps
    noise_data.time_stamps             = timestamps 
     
    return noise_data 

## @ingroup Methods-Noise-Metrics
def SENEL_noise_metric_single_point(SPL):
    """This method calculates the Single Event Noise Exposure Level for a single point (legacy code) 

    Assumptions:
        None

    Source:
        None  
    
    Inputs: 
        SPL      - Noise level 
        
    Outputs: 
        SENEL    - Single Event Noise Exposure Level            [SENEL]
        
    Properties Used:
        N/A     
    """       
    # Maximum noise level on the time history data    
    dBA_max = np.max(SPL)
    
    # Calculates the number of discrete points on the trajectory
    nsteps   = len(SPL)    

    # Finding the time duration for the noise history where the noise level is higher than maximum noise level - 10 dB
    i = 0
    while SPL[i]<=(dBA_max-10) and i<=nsteps:
        i = i+1
    t1 = i # t1 is the first time interval
    i  = i+1

    # Correction for the noise level-10 when it falls outside the limit of the data
    if SPL[nsteps-1]>=(dBA_max-10):
        t2=nsteps-2
    else:
        while i<=nsteps and SPL[i]>=(dBA_max-10):
            i = i+1
        t2 = i-1 #t2 is the last time interval 
        
    # Calculates the integral of the noise which between t1 and t2 points
    sumation = 0
    for i in range (t1-1,t2+1):
        sumation = 10**(SPL[i]/10)+sumation
        
    SENEL = 10*np.log10(sumation)
    
    return SENEL    