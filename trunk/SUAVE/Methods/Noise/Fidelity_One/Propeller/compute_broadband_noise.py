## @ingroupMethods-Noise-Fidelity_One-Propeller
# compute_broadband_noise.py
#
# Created:  Mar 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units , Data 
import numpy as np 
 
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.dbA_noise   import A_weighting  
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools             import SPL_harmonic_to_third_octave

# ----------------------------------------------------------------------
# Frequency Domain Broadband Noise Computation
# ----------------------------------------------------------------------
## @ingroupMethods-Noise-Fidelity_One-Propeller
def compute_broadband_noise(i,freestream,angle_of_attack,position_vector,
                            velocity_vector,propeller,auc_opts,settings,res):
    '''This computes the broadband noise of a propeller or rotor in the frequency domain
    
    Assumptions:
        Coorelated are adjusted from reference altitude of 300 ft 

    Source:
       Schlegel, Ronald, Robert King, and Harold Mull. Helicopter rotor noise generation 
       and propagation. UNITED TECHNOLOGIES CORP STRATFORD CT SIKORSKY AIRCRAFT DIV, 1966.
    
    
    Inputs:
        i                             - control point  
        p_idx                         - index of propeller/rotor 
        freestream                    - freestream data structure
        angle_of_attack               - aircraft angle of attack
        position_vector               - position vector of aircraft
        velocity_vector               - velocity vector of aircraft 
        propeller                     - propeller class data structure
        auc_opts                      - data structure of acoustic data
        settings                      - accoustic settings 
        
        res.      
            SPL_prop_bb_spectrum      - SPL of Frequency Spectrum 
    
    Outputs
       *acoustic data is stored in passed in data structures*
            
    Properties Used:
        N/A   
    '''     
    num_mic        = len(position_vector[:,0,1])
    num_prop       = len(position_vector[0,:,1]) 
    
    x              = position_vector[:,:,0] 
    y              = position_vector[:,:,1]
    z              = position_vector[:,:,2]                                     
    omega          = auc_opts.omega[i]                                      # angular velocity        
    R              = propeller.radius_distribution                          # radial location     
    c              = propeller.chord_distribution                           # blade chord    
    R_tip          = propeller.tip_radius                                   
    beta           = propeller.twist_distribution                           # twist distribution  
    t              = propeller.max_thickness_distribution                   # thickness distribution 
                                                                            
    n              = len(R)                                                 
    S              = np.sqrt(x**2 + y**2 + z**2)                            # distance between rotor and the observer   
    V_tip          = R_tip*omega                                            # blade_tip_speed   
    V_07           = V_tip*0.70/(Units.feet)                                # blade velocity at r/R_tip = 0.7 
    St             = 0.28                                                   # Strouhal number             
    t_avg          = np.mean(t)/(Units.feet)                                # thickness
    c_avg          = np.mean(c)/(Units.feet)                                # average chord  
    beta_07        = beta[round(n*0.70)]                                    # blade angle of attack at r/R = 0.7
    h_val          = t_avg*np.cos(beta_07) + c_avg*np.sin(beta_07)          # projected blade thickness                   
    f_peak         = (V_07*St)/h_val                                        # V - blade velocity at a radial location of 0.7              
    A_blade        = (np.trapz(c, x = R))/(Units.feet**2)                    # area of blade      
    CL_07          = 2*np.pi*beta_07
    S_feet         = S/(Units.feet)
    SPL_300ft      = 10*np.log10(((6.1E-27)*A_blade*V_07**6)/(10**-16)) + 20*np.log(CL_07/0.4)  
    SPL_vals       = SPL_300ft - 20*np.log10(S_feet/300)                       
    
    # estimation of A-Weighting for Vortex Noise  
    f_v            = np.array([0.5*f_peak[0],1*f_peak[0],2*f_peak[0],4*f_peak[0],8*f_peak[0],16*f_peak[0]]) # spectrum
    fr             = f_v/f_peak                                              # frequency ratio  
    weights        = np.atleast_2d(np.array([7.92 , 4.17 , 8.33 , 8.75 ,12.92 , 13.33]))
    SPL_weight     = np.repeat(np.repeat(weights, num_prop, axis = 0)[np.newaxis,:,:], num_mic, axis = 0)    # SPL weight
    SPL_v          = np.repeat(SPL_vals[:,:,np.newaxis], 6 , axis = 2) - SPL_weight            # SPL correction
    dim            = len(f_v)
    C              = np.zeros((num_mic,num_prop,dim))
    p_pref_bb_dBA  = np.zeros((num_mic,num_prop,dim-1))
    SPL_bb_dbAi    = np.zeros((num_mic,num_prop,dim))
    
    for j in range(dim):
        SPL_bb_dbAi[:,:,j] = A_weighting(SPL_v[:,:,j],f_v[j])
    
    for j in range(dim-1):
        C[:,:,j]             = (SPL_bb_dbAi[:,:,j+1] - SPL_bb_dbAi[:,:,j])/(np.log10(fr[j+1]) - np.log10(fr[j])) 
        C[:,:,j+1]           = SPL_bb_dbAi[:,:,j+1] - C[:,:,j]*np.log10(fr[j+1])   
        p_pref_bb_dBA[:,:,j] = (10**(0.1*C[:,:,j+1]))*(((fr[j+1]**(0.1*C[:,:,j]+ 1))/(0.1*C[:,:,j]+ 1))-((fr[j]**(0.1*C[:,:,j]+ 1))/(0.1*C[:,:,j]+ 1))) 
    
    p_pref_bb_dBA[np.isnan(p_pref_bb_dBA)] = 0    
    res.p_pref_bb_dBA  = p_pref_bb_dBA 
     
    # convert to 1/3 octave spectrum   
    res.SPL_prop_bb_spectrum[i] = SPL_harmonic_to_third_octave(SPL_v,f_v,settings)  
    
    return  
 

 
