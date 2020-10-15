# noise_propeller_low_fidelty.py
#
# Created: Oct 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units
import numpy as np 

from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.decibel_arithmetic import SPL_arithmetic
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.dbA_noise          import A_weighting

def compute_total_aircraft_noise(conditions): 
    '''This computes the total aircraft noise by summing up all of the 
    SPLs of the noise sources on the aircraft
    
    ''' 
    angle_of_attack      = conditions.aerodynamics.angle_of_attack 
    ctrl_pts             = len(angle_of_attack)
    num_mic              = conditions.noise.number_of_microphones
    
    # create empty arrays for results  
    num_src          = len(conditions.noise.sources)
    source_SPLs_dBA  = np.zeros((ctrl_pts,num_src,num_mic)) 
    
    # iterate through sources 
    for  i, source in enumerate(conditions.noise.sources.values()):
        
        noise_src = conditions.noise.sources[source]    
        
        if bool(conditions.noise.sources[source]): 
            source_SPLs_dBA[j,i,:] = 0 
        
        else:
            # loop for control points  
            for j in range(ctrl_pts):    
                    
                # collecting unweighted pressure ratios  
                source_SPLs_dBA[j,i,:] = noise_src.SPL_Hv_dBA[j,:]   
    return   
 
