## @ingroupMethods-Noise-Fidelity_One-Propeller
# noise_propeller_low_fidelty.py
#
# Created:  Mar 2021, M. Clarke
# Modified: Jul 2021, E. Botero

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
def propeller_mid_fidelity(network,auc_opts,segment,settings,source = 'propeller'):
    ''' This computes the acoustic signature (sound pressure level, weighted sound pressure levels,
    and frequency spectrums of a system of rotating blades (i.e. propellers and lift_rotors)          
        
    Assumptions:
    None

    Source:
    None
    
    Inputs:
        network                 - vehicle energy network data structure               [None]
        segment                 - flight segment data structure                       [None]
        mic_loc                 - microhone location                                  [m]
        propeller               - propeller class data structure                      [None]
        auc_opts                - data structure of acoustic data                     [None]
        settings                - accoustic settings                                  [None]
                               
    Outputs:
        Results.    
            SPL                 - SPL                                                 [dB]
            SPL_dBA             - dbA-Weighted SPL                                    [dBA]
            SPL_bb_spectrum     - broadband contribution to total SPL                 [dB]
            SPL_spectrum        - 1/3 octave band SPL                                 [dB]
            SPL_tonal_spectrum  - harmonic contribution to total SPL                  [dB]
            SPL_bpfs_spectrum   - 1/3 octave band harmonic contribution to total SPL  [dB]
    
    Properties Used:
        N/A   
    '''
    
    # unpack 
    conditions           = segment.state.conditions
    microphone_locations = conditions.noise.total_microphone_locations
    angle_of_attack      = conditions.aerodynamics.angle_of_attack 
    velocity_vector      = conditions.frames.inertial.velocity_vector
    freestream           = conditions.freestream  
    harmonics            = settings.harmonics  
    
    if not network.identical_propellers:
        assert('This method currently only works with identical propellers')
        
    # Because the propellers are identical, get the first propellers results
    auc_opts = auc_opts[list(auc_opts.keys())[0]]
    
    # create data structures for computation  
    Noise   = Data()  
    Results = Data()
                     
    # compute position vector of microphones         
    position_vector = compute_point_source_coordinates(conditions,network,microphone_locations,source)  
     
    # Harmonic Noise    
    compute_harmonic_noise(harmonics,freestream,angle_of_attack,position_vector,velocity_vector,network,auc_opts,settings,Noise,source)       
     
    # Broadband Noise   
    compute_broadband_noise(freestream,angle_of_attack,position_vector, velocity_vector,network,auc_opts,settings,Noise,source)       
     
    # Combine Rotational(periodic/tonal) and Broadband Noise 
    Noise.SPL_prop_bpfs_spectrum                               = Noise.SPL_r
    Noise.SPL_prop_spectrum                                    = 10*np.log10( 10**(Noise.SPL_prop_h_spectrum/10) + 10**(Noise.SPL_prop_bb_spectrum/10))
    Noise.SPL_prop_spectrum[np.isnan(Noise.SPL_prop_spectrum)] = 0
    
    # pressure ratios used to combine A weighted sound since decibel arithmetic does not work for 
    #broadband noise since it is a continuous spectrum 
    total_p_pref_dBA                                 = np.concatenate((Noise.p_pref_r_dBA,Noise.p_pref_bb_dBA), axis=3)
    Noise.SPL_dBA_prop                               = pressure_ratio_to_SPL_arithmetic(total_p_pref_dBA)  
    Noise.SPL_dBA_prop[np.isinf(Noise.SPL_dBA_prop)] = 0  
    Noise.SPL_dBA_prop[np.isnan(Noise.SPL_dBA_prop)] = 0
    
    # Summation of spectra from propellers into into one SPL
    Results.bpfs                =  Noise.f[:,0,0,0,:] # blade passing frequency harmonics
    Results.SPL                 =  SPL_arithmetic(SPL_arithmetic(Noise.SPL_prop_spectrum))
    Results.SPL_dBA             =  SPL_arithmetic(Noise.SPL_dBA_prop)  
    Results.SPL_spectrum        =  SPL_spectra_arithmetic(Noise.SPL_prop_spectrum)       # 1/3 octave band      
    Results.SPL_bpfs_spectrum   =  SPL_spectra_arithmetic(Noise.SPL_prop_bpfs_spectrum)  # blade passing frequency specturm  
    Results.SPL_tonal_spectrum  =  SPL_spectra_arithmetic(Noise.SPL_prop_tonal_spectrum) 
    Results.SPL_bb_spectrum     =  SPL_spectra_arithmetic(Noise.SPL_prop_bb_spectrum)   
    
    auc_opts.bpfs               =  Results.bpfs               
    auc_opts.SPL                =  Results.SPL                
    auc_opts.SPL_dBA            =  Results.SPL_dBA            
    auc_opts.SPL_spectrum       =  Results.SPL_spectrum       
    auc_opts.SPL_bpfs_spectrum  =  Results.SPL_bpfs_spectrum  
    auc_opts.SPL_tonal_spectrum =  Results.SPL_tonal_spectrum 
    auc_opts.SPL_bb_spectrum    =  Results.SPL_bb_spectrum  
    
    return Results
