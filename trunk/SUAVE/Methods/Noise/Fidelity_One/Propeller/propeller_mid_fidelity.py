## @ingroupMethods-Noise-Fidelity_One-Propeller
# noise_propeller_low_fidelty.py
#
# Created:  Mar 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import  Data
import numpy as np  

from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.decibel_arithmetic import pressure_ratio_to_SPL_arithmetic   
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools                    import SPL_arithmetic
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools                    import SPL_spectra_arithmetic 
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools                    import compute_point_source_coordinates

from SUAVE.Methods.Noise.Fidelity_One.Propeller.compute_broadband_noise  import compute_broadband_noise
from SUAVE.Methods.Noise.Fidelity_One.Propeller.compute_harmonic_noise   import compute_harmonic_noise

# -------------------------------------------------------------------------------------
#  Medium Fidelity Frequency Domain Methods for Acoustic Noise Prediction
# -------------------------------------------------------------------------------------
## @ingroupMethods-Noise-Fidelity_One-Propeller
def propeller_mid_fidelity(network,propeller,auc_opts,segment,settings, mic_loc):
    ''' This computes the acoustic signature (sound pressure level, weighted sound pressure levels,
    and frequency spectrums of a system of rotating blades (i.e. propellers and rotors)          
        
    Assumptions:
    None

    Source:
    None
    
    Inputs:
        network                 - vehicle energy network data structure 
        segment                 - flight segment data structure 
        mic_loc                 - microhone location 
        propeller               - propeller class data structure
        auc_opts                - data structure of acoustic data
        settings                - accoustic settings 
    
    Outputs:
        Results.    
            SPL_tot                 - SPL
            SPL_tot_dBA             - dbA-Weighted SPL 
            SPL_tot_bb_spectrum     - broadband contribution to total SPL
            SPL_tot_spectrum        - 1/3 octave band SPL
            SPL_tot_tonal_spectrum  - harmonic contribution to total SPL
            SPL_tot_bpfs_spectrum   - 1/3 octave band harmonic contribution to total SPL
    
    Properties Used:
        N/A   
    '''
    
    # unpack 
    conditions           = segment.state.conditions
    microphone_locations = conditions.noise.microphone_locations
    angle_of_attack      = conditions.aerodynamics.angle_of_attack 
    velocity_vector      = conditions.frames.inertial.velocity_vector
    freestream           = conditions.freestream 
    N                    = int(network.number_of_engines)                    
    ctrl_pts             = len(angle_of_attack) 
    num_mic              = conditions.noise.number_of_microphones
    num_f                = len(settings.center_frequencies)
    harmonics            = settings.harmonics 
    num_h                = len(harmonics)      
    
    
    # create data structures for computation  
    Noise                             = Data()
    Noise.SPL_dBA_prop                = np.zeros((ctrl_pts,N))  
    Noise.SPL_prop_bb_spectrum        = np.zeros((ctrl_pts,N,num_f))  
    Noise.SPL_prop_spectrum           = np.zeros((ctrl_pts,N,num_f))   
    Noise.SPL_prop_bpfs_spectrum      = np.zeros((ctrl_pts,N,num_f))     
    Noise.SPL_prop_h_spectrum         = np.zeros((ctrl_pts,N,num_f))    
    Noise.SPL_prop_h_dBA_spectrum     = np.zeros((ctrl_pts,N,num_f)) 
    Noise.SPL_prop_tonal_spectrum     = np.zeros((ctrl_pts,N,num_f))     
    Noise.SPL_r                       = np.zeros((N,num_h))  
    Noise.SPL_r_dBA                   = np.zeros_like(Noise.SPL_r)  
    Noise.p_pref_r                    = np.zeros_like(Noise.SPL_r) 
    Noise.p_pref_r_dBA                = np.zeros_like(Noise.SPL_r)     
    Noise.f                           = np.zeros(num_h) 
    
    # create data structures for results
    Results                           = Data()                              
    Results.SPL_tot                   = np.zeros(ctrl_pts) 
    Results.SPL_tot_dBA               = np.zeros(ctrl_pts)  
    Results.SPL_tot_bb_spectrum       = np.zeros((ctrl_pts,num_f))    
    Results.SPL_tot_spectrum          = np.zeros((ctrl_pts,num_f))  
    Results.SPL_tot_tonal_spectrum    = np.zeros((ctrl_pts,num_f))  
    Results.SPL_tot_bpfs_spectrum     = np.zeros((ctrl_pts,num_f))
    
    ## create data structures for computation  
    #Noise                             = Data()
    #Noise.SPL_dBA_prop                = np.zeros((ctrl_pts,num_mic,N))  
    #Noise.SPL_prop_bb_spectrum        = np.zeros((ctrl_pts,num_mic,N,num_f))  
    #Noise.SPL_prop_spectrum           = np.zeros((ctrl_pts,num_mic,N,num_f))   
    #Noise.SPL_prop_bpfs_spectrum      = np.zeros((ctrl_pts,num_mic,N,num_f))     
    #Noise.SPL_prop_h_spectrum         = np.zeros((ctrl_pts,num_mic,N,num_f))    
    #Noise.SPL_prop_h_dBA_spectrum     = np.zeros((ctrl_pts,num_mic,N,num_f)) 
    #Noise.SPL_prop_tonal_spectrum     = np.zeros((ctrl_pts,num_mic,N,num_f))     
    #Noise.SPL_r                       = np.zeros((num_mic,N,num_h))  
    #Noise.SPL_r_dBA                   = np.zeros_like(Noise.SPL_r)  
    #Noise.p_pref_r                    = np.zeros_like(Noise.SPL_r) 
    #Noise.p_pref_r_dBA                = np.zeros_like(Noise.SPL_r)     
    #Noise.f                           = np.zeros(num_h) 
    
    ## create data structures for results
    #Results                           = Data()                              
    #Results.SPL_tot                   = np.zeros((ctrl_pts,num_mic))
    #Results.SPL_tot_dBA               = np.zeros((ctrl_pts,num_mic)) 
    #Results.SPL_tot_bb_spectrum       = np.zeros((ctrl_pts,num_mic,num_f))    
    #Results.SPL_tot_spectrum          = np.zeros((ctrl_pts,num_mic,num_f))  
    #Results.SPL_tot_tonal_spectrum    = np.zeros((ctrl_pts,num_mic,num_f))  
    #Results.SPL_tot_bpfs_spectrum     = np.zeros((ctrl_pts,num_mic,num_f))
                                        
    # loop for control points  
    for i in range(ctrl_pts):    
        
        # loop through number of propellers/rotors 
        for p_idx in range(N):  
            AoA             = angle_of_attack[i][0]   
            thrust_angle    = auc_opts.thrust_angle            
            position_vector = compute_point_source_coordinates(i,p_idx,AoA,thrust_angle,
                                                               microphone_locations,propeller.origin) 
           
            # ------------------------------------------------------------------------------------
            # Harmonic Noise  
            # ------------------------------------------------------------------------------------            
            compute_harmonic_noise(i,num_h,p_idx,harmonics,num_f,freestream,angle_of_attack,
                                   position_vector,velocity_vector,mic_loc,propeller,auc_opts,
                                   settings,Noise)            
            
            # ------------------------------------------------------------------------------------
            # Broadband Noise  
            # ------------------------------------------------------------------------------------ 
            compute_broadband_noise(i ,p_idx ,freestream,angle_of_attack,position_vector,
                                    velocity_vector,mic_loc,propeller,auc_opts,settings,
                                    Noise)       
            
            # ---------------------------------------------------------------------------
            # Combine Rotational(periodic/tonal) and Broadband Noise
            # --------------------------------------------------------------------------- 
            Noise.SPL_prop_bpfs_spectrum[i,p_idx,:num_h] = Noise.SPL_r[p_idx]
            Noise.SPL_prop_spectrum[i,p_idx,:]           = 10*np.log10( 10**(Noise.SPL_prop_h_spectrum[i,p_idx,:]/10) +\
                                                                  10**(Noise.SPL_prop_bb_spectrum[i,p_idx,:]/10))
            
            # pressure ratios used to combine A weighted sound since decibel arithmetic does not work for 
            #broadband noise since it is a continuous spectrum 
            total_p_pref_dBA               = np.concatenate([Noise.p_pref_r_dBA[p_idx,:],Noise.p_pref_bb_dBA])
            Noise.SPL_dBA_prop[i,p_idx]    = pressure_ratio_to_SPL_arithmetic(total_p_pref_dBA)  
            Noise.SPL_dBA_prop[np.isinf(Noise.SPL_dBA_prop)] = 0  
        
        # Summation of spectra from propellers into into one SPL
        Results.SPL_tot[i]                  =  SPL_arithmetic((np.atleast_2d(SPL_arithmetic(Noise.SPL_prop_spectrum[i])))) 
        Results.SPL_tot_dBA[i]              =  SPL_arithmetic((np.atleast_2d(SPL_arithmetic(Noise.SPL_dBA_prop[i]))))      
        Results.SPL_tot_spectrum[i,:]       =  SPL_spectra_arithmetic(Noise.SPL_prop_spectrum[i])       # 1/3 octave band      
        Results.SPL_tot_bpfs_spectrum[i,:]  =  SPL_spectra_arithmetic(Noise.SPL_prop_bpfs_spectrum[i])  # blade passing frequency specturm  
        Results.SPL_tot_tonal_spectrum[i,:] =  SPL_spectra_arithmetic(Noise.SPL_prop_tonal_spectrum[i]) 
        Results.SPL_tot_bb_spectrum[i,:]    =  SPL_spectra_arithmetic(Noise.SPL_prop_bb_spectrum[i])  
     
    return Results
