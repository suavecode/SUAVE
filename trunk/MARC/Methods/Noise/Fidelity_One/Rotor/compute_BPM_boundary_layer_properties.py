## @ingroup Methods-Noise-Fidelity_One-Rotor
# compute_BPM_boundary_layer_properties.py
#
# Created: Feb 2023, M. Clarke 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from MARC.Core import Data   
import numpy as np 

## @ingroup Methods-Noise-Fidelity_One-Rotor  
def compute_BPM_boundary_layer_properties(R_c,c,alpha_star):
    '''This computes the boundary layer properties using the method outlined by the 
    Brooks, Pope and Marcolini (BPM) Model
    
    Assumptions:
        BPM models assumes a naca 0012 airfol  
        
    Source:   
        BPM Model:  Brooks, Thomas F., D. Stuart Pope, and Michael A.
        Marcolini. Airfoil self-noise and prediction. No. L-16528. 1989.
    
    Inputs:  
       a lpha_star - adjustd angle of attack    [deg] 
       c          - airfoil section chord       [m]
       R_c        - Reynolds number             [-]

    Outputs 
       boundary layer properties                [-]
       
    Properties Used:
        N/A   
    '''        
    # eqn 2
    delta_0_div_c_tripped = 10**(1.892- 0.9045*np.log10(R_c) + 0.0596*(np.log10(R_c))**2)
    
    # eqn 3  
    delta_star_0_div_c_tripped             = 0.0601*(R_c**-0.114) #R_c <= 0.3E6
    delta_star_0_div_c_tripped[R_c>0.3E6]  = 10**(3.411 -1.5397*np.log10(R_c[R_c>0.3E6]) + 0.1059*(np.log10(R_c[R_c>0.3E6]))**2) # R_c > 0.3E6 
    
    # eqn 4 
    theta_0_div_c_tripped             = 0.0723*(R_c**-0.1765) #R_c <= 0.3E6
    theta_0_div_c_tripped[R_c>0.3E6]  = 10**(0.5578 -0.7079*np.log10(R_c[R_c>0.3E6]) + 0.0404*(np.log10(R_c[R_c>0.3E6]))**2) # R_c > 0.3E6 
        
    # eqn 5
    delta_0_div_c_untripped       =  10**(1.6569 -0.9045*np.log10(R_c) + 0.0596*(np.log10(R_c))**2) 
    
    # eqn 6
    #delta_star_0_div_c_untripped  =  10**(3.0187 -1.5397*np.log10(R_c) + 0.1059*(np.log10(R_c))**2) 

    # eqn 7
    #theta_0_div_c_untripped       =  10**(0.2021 -0.7079*np.log10(R_c) + 0.0404*(np.log10(R_c))**2) 
    
    # boundary layer of pressure side for tripped and untripped 
    # eqn 8
    #delta_p_div_delta_0_untripped       =  10**(-0.04175*alpha_star + 0.00106*(alpha_star**2))
    #delta_p_div_delta_0_tripped         = delta_p_div_delta_0_untripped
    
    # eqn 9
    delta_star_p_div_delta_0_untripped  =  10**(-0.0432*alpha_star + 0.00113*(alpha_star**2))  
    delta_star_p_div_delta_0_tripped    = delta_star_p_div_delta_0_untripped
    
    # eqn 10
    #theta_p_div_delta_0_untripped       =  10**(-0.04408*alpha_star + 0.000873*(alpha_star**2))  
    #theta_p_div_delta_0_tripped         = theta_p_div_delta_0_untripped
    
    # boundary layer of suction side for tripped and untripped  
    # eqn 11
    delta_s_div_delta_0_tripped                   = 0.3468*(10**(0.1231*alpha_star))
    delta_s_div_delta_0_tripped[alpha_star<5]     = 10**(0.0311*alpha_star[alpha_star<5])
    delta_s_div_delta_0_tripped[alpha_star>12.5]  = 5.718*(10**(0.0258*alpha_star[alpha_star>12.5])) 
     
    # eqn 12  
    delta_star_s_div_delta_0_tripped                   = 0.381*(10**(0.1516*alpha_star))
    delta_star_s_div_delta_0_tripped[alpha_star<5]     = 10**(0.0679*alpha_star[alpha_star<5])
    delta_star_s_div_delta_0_tripped[alpha_star>12.5]  = 14.296*(10**(0.0258*alpha_star[alpha_star>12.5]))   
      
    # eqn 13 
    theta_s_div_delta_0_tripped                   = 0.6984*(10**(0.0869*alpha_star))
    theta_s_div_delta_0_tripped[alpha_star<5]     = 10**(0.0559*alpha_star[alpha_star<5])
    theta_s_div_delta_0_tripped[alpha_star>12.5]  = 4.0846*(10**(0.0258*alpha_star[alpha_star>12.5]))  

    # eqn 14  
    delta_s_div_delta_0_untripped                   = 0.0303*(10**(0.2336*alpha_star))
    delta_s_div_delta_0_untripped[alpha_star<7.5]   = 10**(0.03114*alpha_star[alpha_star<7.5])
    delta_s_div_delta_0_untripped[alpha_star>12.5]  = 12*(10**(0.0258*alpha_star[alpha_star>12.5]))  
     
    # eqn 15 
    delta_star_s_div_delta_0_untripped                   = 0.0162*(10**(0.3066*alpha_star))
    delta_star_s_div_delta_0_untripped[alpha_star<7.5]   = 10**(0.0672*alpha_star[alpha_star<7.5])
    delta_star_s_div_delta_0_untripped[alpha_star>12.5]  = 52.42*(10**(0.0258*alpha_star[alpha_star>12.5]))  
      
    # eqn 16     
    theta_s_div_delta_0_untripped                   = 0.0633*(10**(0.2157*alpha_star))
    theta_s_div_delta_0_untripped[alpha_star<7.5]   = 10**(0.0679*alpha_star[alpha_star<7.5])
    theta_s_div_delta_0_untripped[alpha_star>12.5]  = 14.977*(10**(0.0258*alpha_star[alpha_star>12.5]))   
         
    boundary_layer_data                              = Data() 
    
    # pressure side  
    boundary_layer_data.delta_star_p_untripped       = delta_star_p_div_delta_0_untripped*delta_0_div_c_untripped*c
    boundary_layer_data.delta_star_p_tripped         = delta_star_p_div_delta_0_tripped*delta_0_div_c_tripped*c
    
    # suction side  
    boundary_layer_data.delta_star_s_tripped         = delta_star_s_div_delta_0_tripped*delta_0_div_c_tripped*c
    boundary_layer_data.delta_star_s_untripped       = delta_star_s_div_delta_0_untripped*delta_0_div_c_untripped*c 
    return boundary_layer_data     