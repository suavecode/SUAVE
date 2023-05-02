## @ingroup Methods-Noise-Fidelity_One-Noise_Tools
# dnl_noise.py
# 
# Created: M. Clarke Apr 2023

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from MARC.Core import Units
import numpy as np 
from MARC.Methods.Noise.Fidelity_One.Noise_Tools.decibel_arithmetic   import SPL_arithmetic


# ----------------------------------------------------------------------
#  dbl Noise
# ----------------------------------------------------------------------

## @ingroup Methods-Noise-Fidelity_One-Noise_Tools
def dnl_noise(noise_data, flight_times = ['06:00:00'],time_period = 24*Units.hours):  

    """This method calculates the Day-Night noise level between a given time 
    
    Assumptions:
        Flights occure between 6:00 and 9:00 pm 

    Source:
        None

    Inputs:
       noise_data  - post-processed noise data structure 

    Outputs: [dB]
       noise_data  - post-processed noise data structure 

    Properties Used:
        N/A  
    """    
     

    SPL_dbA    = noise_data.SPL_dBA       
    t          = noise_data.time    
    N_gm_y     = noise_data.N_gm_y                 
    N_gm_x     = noise_data.N_gm_x                 
    time_step  = t[1]-t[0]
    
    # Compute Day-Night Sound Level and Noise Equivalent Noise   
    number_of_flights       = len(flight_times) 
    T                       = 15*Units.hours
    total_time_steps        = int(T/time_step) 
 
    CME = np.zeros((total_time_steps,N_gm_x,N_gm_y)) # temporal noise exposure
    for i in range(number_of_flights): 
        # get start time of flight
        t0  = int((np.float(flight_times[i].split(':')[0])*60*60 + \
                  np.float(flight_times[i].split(':')[1])*60 + \
                  np.float(flight_times[i].split(':')[2]) - 6*Units.hours)/time_step)    
        CME[t0:t0+len(t)] = SPL_arithmetic(np.concatenate((CME[t0:t0+len(t)][:,:,:,None] , SPL_dbA[:,:,:,None]), axis=3), sum_axis=3) 
    
    # DNL Noise 
    idx_7am        = int(1*Units.hours/time_step)  
    CME[0:idx_7am] = CME[0:idx_7am] + 10 
    delta_t        = time_step*np.ones((total_time_steps,N_gm_x,N_gm_y)) 
    p_dn_i         = 10**(CME/10)   
    L_dn           = 10*np.log10((1/T)*np.sum(p_dn_i*delta_t, axis = 0)) 
    
    return L_dn
