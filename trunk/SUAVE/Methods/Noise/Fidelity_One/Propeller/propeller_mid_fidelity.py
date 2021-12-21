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

from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.decibel_arithmetic           import pressure_ratio_to_SPL_arithmetic   
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.decibel_arithmetic           import SPL_arithmetic
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.decibel_arithmetic           import SPL_spectra_arithmetic  
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.compute_source_coordinates   import compute_point_source_coordinates
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.compute_source_coordinates   import compute_blade_section_source_coordinates
from SUAVE.Methods.Noise.Fidelity_One.Propeller.compute_harmonic_noise         import compute_harmonic_noise
from SUAVE.Methods.Noise.Fidelity_One.Propeller.compute_broadband_noise_old    import compute_broadband_noise_old 
from SUAVE.Methods.Noise.Fidelity_One.Propeller.compute_broadband_noise        import compute_broadband_noise

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
            blade_passing_frequencies      - blade passing frequencies                           [Hz]
            SPL                            - total SPL                                           [dB]
            SPL_dBA                        - dbA-Weighted SPL                                    [dBA]
            SPL_1_3_spectrum               - 1/3 octave band spectrum of SPL                     [dB]
            SPL_1_3_spectrum_dBA           - 1/3 octave band spectrum of A-weighted SPL          [dBA]
            SPL_broadband_1_3_spectrum     - 1/3 octave band broadband contribution to total SPL [dB] 
            SPL_harmonic_1_3_spectrum      - 1/3 octave band harmonic contribution to total SPL  [dB]
            SPL_harmonic_bpf_spectrum_dBA  - A-weighted blade passing freqency spectrum of 
                                             harmonic compoment of SPL                           [dB]
            SPL_harmonic_bpf_spectrum      - blade passing freqency spectrum of harmonic
                                             compoment of SPL                                    [dB] 
     
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
                     
     # compute position vector from point source at rotor hub to microphones  
    position_vector = compute_point_source_coordinates(conditions,network,microphone_locations,source,settings)  
     
    # Harmonic Noise    
    compute_harmonic_noise(harmonics,freestream,angle_of_attack,position_vector,velocity_vector,network,auc_opts,settings,Noise,source)       
    
    # compute position vector of blade section source to microphones   
    blade_section_position_vectors = compute_blade_section_source_coordinates(angle_of_attack,auc_opts,network,microphone_locations,source,settings)   
    
    # Broadband Noise   
    compute_broadband_noise_old(freestream,angle_of_attack,blade_section_position_vectors,velocity_vector,network,auc_opts,settings,Noise,source)  
    #compute_broadband_noise(freestream,angle_of_attack,blade_section_position_vectors,velocity_vector,network,auc_opts,settings,Noise,source)        
     
    # Combine Harmonic (periodic/tonal) and Broadband Noise  
    #Noise.SPL_total_1_3_spectrum  = 10*np.log10( 10**(Noise.SPL_prop_harmonic_1_3_spectrum/10) + 10**(Noise.SPL_prop_broadband_1_3_spectrum/10))
    Noise.SPL_total_1_3_spectrum  = 10*np.log10( 10**(Noise.SPL_prop_harmonic_1_3_spectrum/10)) # ONLY HARMONIC
    Noise.SPL_total_1_3_spectrum[np.isnan(Noise.SPL_total_1_3_spectrum)] = 0
    #Noise.SPL_total_1_3_spectrum_dBA  = 10*np.log10( 10**(Noise.SPL_prop_harmonic_1_3_spectrum_dBA/10) + 10**(Noise.SPL_prop_broadband_1_3_spectrum_dBA/10))
    Noise.SPL_total_1_3_spectrum_dBA  = 10*np.log10( 10**(Noise.SPL_prop_harmonic_1_3_spectrum_dBA/10))
    Noise.SPL_total_1_3_spectrum_dBA[np.isnan(Noise.SPL_total_1_3_spectrum)] = 0
    
    # pressure ratios used to combine A weighted sound since decibel arithmetic does not work for 
    #broadband noise since it is a continuous spectrum 
    #total_p_pref_dBA                                  = np.concatenate((Noise.p_pref_harmonic_dBA,Noise.p_pref_broadband_dBA), axis=3) 
    total_p_pref_dBA                                  = Noise.p_pref_harmonic_dBA
    Noise.SPL_total_dBA                               = pressure_ratio_to_SPL_arithmetic(total_p_pref_dBA)  
    Noise.SPL_total_dBA[np.isinf(Noise.SPL_total_dBA)] = 0  
    Noise.SPL_total_dBA[np.isnan(Noise.SPL_total_dBA)] = 0
    
    # Summation of spectra from propellers into into one SPL and store results
    Results.blade_passing_frequencies                     = Noise.f[:,0,0,0,:]       
    Results.SPL                                           = SPL_spectra_arithmetic(SPL_spectra_arithmetic(Noise.SPL_total_1_3_spectrum))       
    Results.SPL_harmonic                                  = SPL_spectra_arithmetic(SPL_spectra_arithmetic(Noise.SPL_prop_harmonic_1_3_spectrum))    
    #Results.SPL_broadband                                 = SPL_spectra_arithmetic(SPL_spectra_arithmetic(Noise.SPL_prop_broadband_1_3_spectrum)) 
    Results.SPL_dBA                                       = SPL_spectra_arithmetic(SPL_spectra_arithmetic(Noise.SPL_total_1_3_spectrum_dBA))     
    Results.SPL_harmonic_dBA                              = SPL_spectra_arithmetic(SPL_spectra_arithmetic(Noise.SPL_prop_harmonic_1_3_spectrum_dBA))    
    #Results.SPL_broadband_dBA                             = SPL_spectra_arithmetic(SPL_spectra_arithmetic(Noise.SPL_prop_broadband_1_3_spectrum_dBA)) 
    Results.SPL_harmonic_bpf_spectrum_dBA                 = SPL_spectra_arithmetic(Noise.SPL_prop_harmonic_bpf_spectrum_dBA )  
    Results.SPL_harmonic_bpf_spectrum                     = SPL_spectra_arithmetic(Noise.SPL_prop_harmonic_bpf_spectrum ) 
    Results.SPL_1_3_spectrum                              = SPL_spectra_arithmetic(Noise.SPL_total_1_3_spectrum)      
    Results.SPL_1_3_spectrum_dBA                          = SPL_spectra_arithmetic(Noise.SPL_total_1_3_spectrum_dBA)      
    Results.SPL_harmonic_1_3_spectrum                     = SPL_spectra_arithmetic(Noise.SPL_prop_harmonic_1_3_spectrum)       
    #Results.SPL_broadband_1_3_spectrum                    = SPL_spectra_arithmetic(Noise.SPL_prop_broadband_1_3_spectrum)  
    Results.one_third_frequency_spectrum                  = settings.center_frequencies 
    #Results.SPL_prop_azimuthal_harmonic                   = Noise.p_harmonic  
    #Results.azimuthal_time                                = Noise.azimuthal_time
    #Results.p_pref_azimuthal_broadband                    = Noise.p_pref_azimuthal_broadband                    
    #Results.p_pref_azimuthal_broadband_dBA                = Noise.p_pref_azimuthal_broadband_dBA                
    #Results.SPL_prop_azimuthal_broadband_spectrum         = Noise.SPL_prop_azimuthal_broadband_spectrum         
    #Results.SPL_prop_azimuthal_broadband_spectrum_dBA     = Noise.SPL_prop_azimuthal_broadband_spectrum_dBA     
    
    return Results
