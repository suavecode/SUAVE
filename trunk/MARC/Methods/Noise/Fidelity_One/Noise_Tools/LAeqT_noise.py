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
def LAeqT_noise(noise_data, flight_times = ['06:00:00'],time_period = 24*Units.hours):  

    """This method calculates the Average A-weighted Sound Level, LAeqT,
    also known as the equivalent continuous noise level
    
    Assumptions:  

    Source:
        None

    Inputs:
       noise_data  - post-processed noise data structure 

    Outputs: [dB]
       noise_data  - post-processed noise data structure 

    Properties Used:
        N/A  
    """    
     

    SPL_dbA    = noise_data.SPL_dBA_ground_mic       
    t          = noise_data.time    
    N_gm_y     = noise_data.N_gm_y                 
    N_gm_x     = noise_data.N_gm_x                 
     
    # Compute Day-Night Sound Level and Noise Equivalent Noise   
    number_of_flights       = len(flight_times) 
    total_time_steps        = int(15*Units.hours/time_step) 
 
    TNE = np.zeros((total_time_steps,N_gm_x,N_gm_y))  # temporal noise exposure
    for i in range(number_of_flights): 
        # get start time of flight
        t0  = int((np.float(flight_times[i].split(':')[0])*60*60 + \
                  np.float(flight_times[i].split(':')[1])*60 + \
                  np.float(flight_times[i].split(':')[2]) - 6*Units.hours)/time_step)    
        TNE[t0:t0+len(t)] = SPL_arithmetic(np.concatenate((TNE[t0:t0+len(t)][:,:,:,None] , SPL_dbA[:,:,:,None]), axis=3), sum_axis=3) 
   
    # Equivalent noise 
    delta_t  = time_step*np.ones((total_time_steps,N_gm_x,N_gm_y))  
    p_i      = 10**(TNE/10)  
    LAeqT    = 10*np.log10(np.sum(p_i*delta_t, axis = 0)) 
    
    return LAeqT
