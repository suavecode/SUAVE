## @ingroup Methods-Noise-Fidelity_One-Propeller
# noise_propeller_low_fidelty.py
#
# Created:  Mar 2021, M. Clarke
# Modified: Jul 2021, E. Botero
#           Feb 2022, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
from MARC.Core import  Data 
import numpy as np   
from MARC.Methods.Noise.Fidelity_One.Noise_Tools.decibel_arithmetic                 import SPL_arithmetic  
from MARC.Methods.Noise.Fidelity_One.Noise_Tools.compute_noise_source_coordinates   import compute_rotor_point_source_coordinates
from MARC.Methods.Noise.Fidelity_One.Rotor.compute_harmonic_noise                   import compute_harmonic_noise
from MARC.Methods.Noise.Fidelity_One.Rotor.compute_broadband_noise                  import compute_broadband_noise

# -------------------------------------------------------------------------------------
#  Medium Fidelity Frequency Domain Methods for Acoustic Noise Prediction
# -------------------------------------------------------------------------------------
## @ingroup Methods-Noise-Fidelity_One-Propeller
def total_rotor_noise(rotors,aeroacoustic_data,segment,settings):
    ''' This computes the acoustic signature (sound pressure level, weighted sound pressure levels,
    and frequency spectrums of a system of rotating blades           
        
    Assumptions:
    None

    Source:
    None
    
    Inputs:
        rotors                  - data structure of rotors                            [None]
        segment                 - flight segment data structure                       [None] 
        aeroacoustic_data       - data structure of acoustic data                     [None]
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
    
    # create data structures for computation
    Noise   = Data()  
    Results = Data()
                     
     # compute position vector from point source at rotor hub to microphones
    coordinates = compute_rotor_point_source_coordinates(conditions,rotors,microphone_locations,settings) 

    # Harmonic Noise    
    compute_harmonic_noise(harmonics,freestream,angle_of_attack,coordinates,velocity_vector,rotors,aeroacoustic_data,settings,Noise)       
     
    # Broadband Noise
    compute_broadband_noise(freestream,angle_of_attack,coordinates,velocity_vector,rotors,aeroacoustic_data,settings,Noise)

    # Combine Harmonic (periodic/tonal) and Broadband Noise
    Noise.SPL_total_1_3_spectrum      = 10*np.log10( 10**(Noise.SPL_prop_harmonic_1_3_spectrum/10) + 10**(Noise.SPL_prop_broadband_1_3_spectrum/10)) 
    Noise.SPL_total_1_3_spectrum[np.isnan(Noise.SPL_total_1_3_spectrum)] = 0
    Noise.SPL_total_1_3_spectrum_dBA  = 10*np.log10( 10**(Noise.SPL_prop_harmonic_1_3_spectrum_dBA/10) + 10**(Noise.SPL_prop_broadband_1_3_spectrum_dBA/10))
    Noise.SPL_total_1_3_spectrum_dBA[np.isnan(Noise.SPL_total_1_3_spectrum)] = 0

    # Summation of spectra from propellers into into one SPL and store results
    Results.blade_passing_frequencies                     = Noise.f[:,0,0,0,:]       
    Results.SPL                                           = SPL_arithmetic(SPL_arithmetic(Noise.SPL_total_1_3_spectrum))       
    Results.SPL_dBA                                       = SPL_arithmetic(SPL_arithmetic(Noise.SPL_total_1_3_spectrum_dBA))     
    Results.SPL_harmonic_1_3_spectrum_dBA                 = SPL_arithmetic(Noise.SPL_prop_harmonic_1_3_spectrum_dBA)   
    Results.SPL_broadband_1_3_spectrum_dBA                = SPL_arithmetic(Noise.SPL_prop_broadband_1_3_spectrum_dBA)
    Results.SPL_harmonic_bpf_spectrum                     = SPL_arithmetic(Noise.SPL_prop_harmonic_bpf_spectrum ) 
    Results.SPL_1_3_spectrum                              = SPL_arithmetic(Noise.SPL_total_1_3_spectrum)      
    Results.SPL_1_3_spectrum_dBA                          = SPL_arithmetic(Noise.SPL_total_1_3_spectrum_dBA)      
    Results.one_third_frequency_spectrum                  = settings.center_frequencies 
    Results.SPL_harmonic_bpf_spectrum_dBA                 = SPL_arithmetic(Noise.SPL_prop_harmonic_bpf_spectrum_dBA )  
    Results.SPL_harmonic                                  = SPL_arithmetic(SPL_arithmetic(Noise.SPL_prop_harmonic_1_3_spectrum))    
    Results.SPL_broadband                                 = SPL_arithmetic(SPL_arithmetic(Noise.SPL_prop_broadband_1_3_spectrum)) 
    Results.SPL_harmonic_1_3_spectrum                     = SPL_arithmetic(Noise.SPL_prop_harmonic_1_3_spectrum)       
    Results.SPL_broadband_1_3_spectrum                    = SPL_arithmetic(Noise.SPL_prop_broadband_1_3_spectrum) 
    
    return Results
