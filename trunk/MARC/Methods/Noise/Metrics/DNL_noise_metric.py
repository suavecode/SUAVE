## @ingroup Methods-Noise-Metrics
# DNL_noise_metric.py
# 
# Created: M. Clarke Apr 2023

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from MARC.Core import Units
import numpy as np 
from MARC.Methods.Noise.Common.decibel_arithmetic   import SPL_arithmetic
from MARC.Methods.Noise.Common.background_noise     import background_noise


# ----------------------------------------------------------------------
#  dbl Noise
# ----------------------------------------------------------------------

## @ingroup Methods-Noise-Metrics
def DNL_noise_metric(noise_data, flight_times = ['12:00:00'],time_period = 24*Units.hours):  

    """This method calculates the Day-Night noise level between a given time 
    
    Assumptions:
        Flights occure between 6:00 and 9:00 pm  (i.e. a 15 hour window)

    Source:
        None

    Inputs:
       noise_data  - post-processed noise data structure 

    Outputs: [dB]
       noise_data  - post-processed noise data structure 

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
        number_of_timesteps        = int(T/time_step) 
        timestamps              = np.linspace(0,T,number_of_timesteps)
     
        TNE = np.zeros((number_of_timesteps,N_gm_x,N_gm_y))*np.nan   # cumulative noise exposure
        SPL[SPL == background_noise()] = np.nan
        for i in range(number_of_flights): 
            # get start time of flight
            t0  = int((np.float(flight_times[i].split(':')[0])*60*60 + \
                      np.float(flight_times[i].split(':')[1])*60 + \
                      np.float(flight_times[i].split(':')[2]) - 6*Units.hours)/time_step)    
            p_prefs_A               = 10**(TNE[t0:t0+len(t)][:,:,:,None]/10)
            p_prefs_B               = 10**(SPL[:,:,:,None]/10)
            C                       = np.concatenate((p_prefs_A,p_prefs_B),axis = 3)
            TNE[t0:t0+len(t)]       = 10*np.log10(np.nansum(C,axis=3))  
            TNE[t0:t0+len(t)]       = SPL_arithmetic(np.concatenate((TNE[t0:t0+len(t)][:,:,:,None] , SPL[:,:,:,None]), axis=3), sum_axis=3) 
            
    
    else:
        time_step           = noise_data.time_step            
        TNE                 = noise_data.temporal_noise_exposure 
        number_of_timesteps = noise_data.number_of_timesteps
        timestamps          = noise_data.time_stamps
        N_gm_x              = len(TNE[0,:,0])
        N_gm_y              = len(TNE[0,0,:])
    
    # Day-Night Average Noise Level
    delta_t                 = time_step*np.ones((number_of_timesteps,N_gm_x,N_gm_y))  
    idx_7am                 = int(1*Units.hours/time_step)  
    L_dn                    = np.zeros_like(TNE)
    L_dn[:,:,:]             = TNE
    L_dn[0:idx_7am]         = L_dn[0:idx_7am] + 10   
    p_dn_i                  = 10**(L_dn/10)    
    DNL                     = 10*np.log10((1/(24*Units.hours))*np.nansum(p_dn_i*delta_t, axis = 0))   
    #DNL[DNL == np.nan] = 35.
    #DNL[DNL < 35] = 35.
    #np.nan_to_num(DNL)    
      
    noise_data.DNL                     = DNL
    noise_data.temporal_noise_exposure = TNE 
    noise_data.time_step               = time_step
    noise_data.time_stamps             = timestamps 
    noise_data.number_of_timesteps     = number_of_timesteps
    
    return noise_data 
