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
def compute_broadband_noise(freestream,angle_of_attack,position_vector,
                            velocity_vector,network,auc_opts,settings,res):
    '''This computes the broadband noise of a propeller or rotor in the frequency domain
    
    Assumptions:
        Coorelations are adjusted from reference altitude of 300 ft 

    Source:
       Schlegel, Ronald, Robert King, and Harold Mull. Helicopter rotor noise generation 
       and propagation. UNITED TECHNOLOGIES CORP STRATFORD CT SIKORSKY AIRCRAFT DIV, 1966.
    
    
    Inputs:  
        freestream                    - freestream data structure        [m/s]
        angle_of_attack               - aircraft angle of attack         [rad]
        position_vector               - position vector of aircraft      [m]
        velocity_vector               - velocity vector of aircraft      [m/s]
        network                       - energy network object            [None] 
        auc_opts                      - data structure of acoustic data  [None] 
        settings                      - accoustic settings               [None]
        res.      
            SPL_prop_bb_spectrum      - SPL of Frequency Spectrum        [dB]
    
    Outputs
       *acoustic data is stored and passed in data structures*
            
    Properties Used:
        N/A   
    '''     
    num_cpt        = len(angle_of_attack)
    num_mic        = len(position_vector[0,:,0,1])
    num_prop       = len(position_vector[0,0,:,1])
    propellers     = network.propellers
    propeller      = network.propellers[list(propellers.keys())[0]]
    
    # ----------------------------------------------------------------------------------
    # Broadband (Vortex) Noise
    # ----------------------------------------------------------------------------------      
    # [number control points, number of mics, number of props, positions]
    
    x              = position_vector[:,:,:,0]                               
    y              = position_vector[:,:,:,1]
    z              = position_vector[:,:,:,2]                                     
    omega          = auc_opts.omega[:,0]                                    # angular velocity        
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
    
    SPL_300ft      = np.atleast_2d(10*np.log10(((6.1E-27)*A_blade*V_07**6)/(10**-16)) + 20*np.log(CL_07/0.4)) 
    SPL_300ft      = np.repeat(np.repeat(SPL_300ft.T,num_mic, axis = 1)[:,:,np.newaxis],num_prop, axis = 2)
    SPL_vals       = SPL_300ft - 20*np.log10(S_feet/300)                       
    SPL_vals       = SPL_300ft - 20*np.log10(S_feet/300)                       
    
    # estimation of A-Weighting for Vortex Noise  
    f_v            = np.array([0.5*f_peak,1*f_peak,2*f_peak,4*f_peak,8*f_peak,16*f_peak]).T # spectrum
    fr             = np.array([0.5,1,2,4,8,16])                                           # frequency ratio  
    weights        = np.atleast_2d(np.array([7.92 , 4.17 , 8.33 , 8.75 ,12.92 , 13.33]))
    SPL_weight     = np.repeat(np.repeat(np.repeat(weights, num_prop, axis = 0)[np.newaxis,:,:], num_mic, axis = 0)[np.newaxis,:,:,:], num_cpt, axis = 0)    # SPL weight
    SPL_v          = np.repeat(SPL_vals[:,:,:,np.newaxis], 6 , axis = 3) - SPL_weight                          # SPL correction 
    dim            = len(fr)
    C              = np.zeros((num_cpt,num_mic,num_prop,dim))
    p_pref_bb_dBA  = np.zeros((num_cpt,num_mic,num_prop,dim-1))
    SPL_bb_dbAi    = np.zeros((num_cpt,num_mic,num_prop,dim))
    
    for i in range(num_cpt): 
        for j in range(dim):
            SPL_bb_dbAi[i,:,:,j] = A_weighting(SPL_v[i,:,:,j],f_v[i,j])
    
    for j in range(dim-1):
        C[:,:,:,j]             = (SPL_bb_dbAi[:,:,:,j+1] - SPL_bb_dbAi[:,:,:,j])/(np.log10(fr[j+1]) - np.log10(fr[j])) 
        C[:,:,:,j+1]           = SPL_bb_dbAi[:,:,:,j+1] - C[:,:,:,j]*np.log10(fr[j+1])   
        p_pref_bb_dBA[:,:,:,j] = (10**(0.1*C[:,:,:,j+1]))*(((fr[j+1]**(0.1*C[:,:,:,j]+ 1))/(0.1*C[:,:,:,j]+ 1))-((fr[j]**(0.1*C[:,:,:,j]+ 1))/(0.1*C[:,:,:,j]+ 1))) 
    
    p_pref_bb_dBA[np.isnan(p_pref_bb_dBA)] = 0    
    res.p_pref_bb_dBA  = p_pref_bb_dBA 
     
    # convert to 1/3 octave spectrum   
    res.SPL_prop_bb_spectrum = SPL_harmonic_to_third_octave(SPL_v,f_v,settings)  
    
    return  
 

 
