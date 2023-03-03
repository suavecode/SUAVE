## @ingroup Methods-Noise-Fidelity_One-Rotor
# compute_broadband_noise.py
#
# Created:   Mar 2021, M. Clarke
# Modified:  Feb 2022, M. Clarke
# Modified:  Sep 2022, M. Clarke
# Modified:  Sep 2022, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np  
from MARC.Core import Units
from MARC.Methods.Noise.Fidelity_One.Noise_Tools.dbA_noise                     import A_weighting
from MARC.Methods.Noise.Fidelity_One.Noise_Tools.convert_to_third_octave_band  import convert_to_third_octave_band  
from MARC.Methods.Noise.Fidelity_One.Noise_Tools.decibel_arithmetic            import SPL_arithmetic 

from MARC.Methods.Noise.Fidelity_One.Rotor.compute_BPM_boundary_layer_properties import compute_BPM_boundary_layer_properties
from MARC.Methods.Noise.Fidelity_One.Rotor.compute_LBL_VS_broadband_noise        import compute_LBL_VS_broadband_noise
from MARC.Methods.Noise.Fidelity_One.Rotor.compute_TBL_TE_broadband_noise        import compute_TBL_TE_broadband_noise
from MARC.Methods.Noise.Fidelity_One.Rotor.compute_TIP_broadband_noise           import compute_TIP_broadband_noise 
from MARC.Methods.Noise.Fidelity_One.Rotor.compute_noise_directivities           import compute_noise_directivities
 
# ----------------------------------------------------------------------
# Frequency Domain Broadband Noise Computation
# ----------------------------------------------------------------------

## @ingroup Methods-Noise-Fidelity_One-Propeller   
def compute_broadband_noise(freestream,angle_of_attack,coordinates,
                            velocity_vector,rotors,aeroacoustic_data,settings,res):
    '''This computes the trailing edge noise compoment of broadband noise of a propeller or 
    lift-rotor in the frequency domain. Boundary layer properties are computed using MARC's 
    panel method.
    
    Assumptions:
        Boundary layer thickness (delta) appear to be an order of magnitude off at the trailing edge and 
        correction factor of 0.1 is used. See lines 255 and 256 
        
    Source: 
        Li, Sicheng Kevin, and Seongkyu Lee. "Prediction of Urban Air Mobility Multirotor VTOL Broadband Noise
        Using UCD-QuietFly." Journal of the American Helicopter Society (2021).
    
    Inputs:  
        freestream                                   - freestream data structure                                                          [m/s]
        angle_of_attack                              - aircraft angle of attack                                                           [rad]
        bspv                                         - rotor blade section trailing position edge vectors                                 [m]
        velocity_vector                              - velocity vector of aircraft                                                        [m/s]
        rotors                                       - data structure of rotors                                                           [None] 
        aeroacoustic_data                            - data structure of acoustic data                                                    [None] 
        settings                                     - accoustic settings                                                                 [None] 
        res                                          - results data structure                                                             [None] 
    
    Outputs 
       res.                                           *acoustic data is stored and passed in data structures*                                          
           SPL_prop_broadband_spectrum               - broadband noise in blade passing frequency spectrum                                [dB]
           SPL_prop_broadband_spectrum_dBA           - dBA-Weighted broadband noise in blade passing frequency spectrum                   [dbA]     
           SPL_prop_broadband_1_3_spectrum           - broadband noise in 1/3 octave spectrum                                             [dB]
           SPL_prop_broadband_1_3_spectrum_dBA       - dBA-Weighted broadband noise in 1/3 octave spectrum                                [dBA]                               
           p_pref_azimuthal_broadband                - azimuthal varying pressure ratio of broadband noise                                [Unitless]       
           p_pref_azimuthal_broadband_dBA            - azimuthal varying pressure ratio of dBA-weighted broadband noise                   [Unitless]     
           SPL_prop_azimuthal_broadband_spectrum     - azimuthal varying broadband noise in blade passing frequency spectrum              [dB]      
           SPL_prop_azimuthal_broadband_spectrum_dBA - azimuthal varying dBA-Weighted broadband noise in blade passing frequency spectrum [dbA]   
        
    Properties Used:
        N/A   
    '''     

    num_cpt       = len(coordinates.X[:,0,0,0,0,0])
    num_mic       = len(coordinates.X[0,:,0,0,0,0])  
    num_rot       = len(coordinates.X[0,0,:,0,0,0])
    num_blades    = len(coordinates.X[0,0,0,:,0,0])
    num_sec       = len(coordinates.X[0,0,0,0,:,0])
    rotor         = rotors[list(rotors.keys())[0]]
    frequency     = settings.center_frequencies
    num_cf        = len(frequency)     
    
    # ----------------------------------------------------------------------------------
    # Trailing Edge Noise
    # ---------------------------------------------------------------------------------- 
    p_ref              = 2E-5                               # referece atmospheric pressure
    speed_of_sound     = freestream.speed_of_sound          # speed of sound
    density            = freestream.density                 # air density 
    dyna_visc          = freestream.dynamic_viscosity  
    alpha              = aeroacoustic_data.blade_effective_angle_of_attack  
    blade_Re           = aeroacoustic_data.blade_reynolds_number
    blade_speed        = aeroacoustic_data.blade_velocity
    Vt                 = aeroacoustic_data.blade_tangential_velocity  
    Va                 = aeroacoustic_data.blade_axial_velocity                
    blade_chords       = rotor.chord_distribution           # blade chord    
    r                  = rotor.radius_distribution          # radial location   
    Omega              = aeroacoustic_data.omega            # angular velocity    
    L                  = np.zeros_like(r)
    del_r              = r[1:] - r[:-1]
    L[0]               = del_r[0]
    L[-1]              = del_r[-1]
    L[1:-1]            = (del_r[:-1]+ del_r[1:])/2
 

    if np.all(Omega == 0):
        res.p_pref_broadband                          = np.zeros((num_cpt,num_mic,num_rot,num_cf)) 
        res.SPL_prop_broadband_spectrum               = np.zeros_like(res.p_pref_broadband)
        res.SPL_prop_broadband_spectrum_dBA           = np.zeros_like(res.p_pref_broadband)
        res.SPL_prop_broadband_1_3_spectrum           = np.zeros((num_cpt,num_mic,num_rot,num_cf))
        res.SPL_prop_broadband_1_3_spectrum_dBA       = np.zeros((num_cpt,num_mic,num_rot,num_cf))
        res.p_pref_azimuthal_broadband                = np.zeros((num_cpt,num_mic,num_rot,num_cf))
        res.p_pref_azimuthal_broadband_dBA            = np.zeros_like(res.p_pref_azimuthal_broadband)
        res.SPL_prop_azimuthal_broadband_spectrum     = np.zeros_like(res.p_pref_azimuthal_broadband)
        res.SPL_prop_azimuthal_broadband_spectrum_dBA = np.zeros_like(res.p_pref_azimuthal_broadband)
    else:
        
        # dimension of matrices [control point, microphone , num rotors , number of blades, number of sections ,num center frequencies ] 
        c                 = np.tile(blade_chords[None,None,None,None,:,None],(num_cpt,num_mic,num_rot,num_blades,1,num_cf))
        L                  = np.tile(L[None,None,None,None,:,None],(num_cpt,num_mic,num_rot,num_blades,1,num_cf))
        f                 = np.tile(frequency[None,None,None,None,None,:],(num_cpt,num_mic,num_rot,num_blades,num_sec,1))
        alpha_blade       = np.tile(alpha[:,None,None,None,:,None],(1,num_mic,num_rot,num_blades,1,num_cf))
        V                 = np.zeros((num_cpt,num_mic,num_rot,num_blades,num_sec,num_cf,3))
        V[:,:,:,:,:,:,0]  = -np.tile(Vt[:,None,None,None,:,None],(1,num_mic,num_rot,num_blades,1,num_cf)) 
        V[:,:,:,:,:,:,2]  = np.tile(Va[:,None,None,None,:,None],(1,num_mic,num_rot,num_blades,1,num_cf))  
        V_tot             = np.linalg.norm(V,axis= 6)
        #U_inf             = np.tile(velocity_vector[:,:,None,None,None,None,None],(1,1,num_mic,num_rot,num_blades,num_sec,num_cf)) 
        c_0               = np.tile(speed_of_sound[:,:,None,None,None,None],(1,num_mic,num_rot,num_blades,num_sec,num_cf))
        rho               = np.tile(density[:,:,None,None,None,None],(1,num_mic,num_rot,num_blades,num_sec,num_cf)) 
        mu                = np.tile(dyna_visc[:,:,None,None,None,None],(1,num_mic,num_rot,num_blades,num_sec,num_cf)) 
        R_c               = np.tile(blade_Re[:,None,None,None,:,None],(1,num_mic,num_rot,num_blades,1,num_cf))
        U                 = np.tile(blade_speed[:,None,None,None,:,None],(1,num_mic,num_rot,num_blades,1,num_cf))
        M                 = U/c_0 
        M_tot             = V_tot/c_0   
          
        X_prime_r         = np.tile(coordinates.X_prime_r[:,:,:,:,:,None,:],(1,1,1,1,1,num_cf,1))
        cos_zeta_r        = np.sum(X_prime_r*V, axis = 6)/(np.linalg.norm(X_prime_r, axis = 6)*np.linalg.norm(V, axis = 6) )                 
        norm_X_r          = np.tile(np.linalg.norm(coordinates.X_r,axis = 5)[:,:,:,:,:,None],(1,1,1,1,1,num_cf)) 
        sigma             = np.sqrt((norm_X_r**2)*((1 - M_tot*cos_zeta_r)**2))  
        Y_er              = np.tile(np.sqrt(coordinates.X_e_r[:,:,:,:,:,1]**2 +  coordinates.X_e_r[:,:,:,:,:,2]**2)[:,:,:,:,:,None],(1,1,1,1,1,num_cf))                      
        r_er              = np.tile(np.linalg.norm(coordinates.X_e_r, axis = 5)[:,:,:,:,:,None],(1,1,1,1,1,num_cf))           
        Phi_er            = np.arcsin(np.tile(coordinates.X_e_r[:,:,:,:,:,1,None],(1,1,1,1,1,num_cf))/Y_er)  
        Theta_er          = np.tile(coordinates.theta_e_r[:,:,:,:,:,None],(1,1,1,1,1,num_cf))    
        
        # flatten matrices 
        R_c        = flatten_matrix(R_c,num_cpt,num_mic,num_rot,num_blades,num_sec,num_cf)
        c          = flatten_matrix(c,num_cpt,num_mic,num_rot,num_blades,num_sec,num_cf)
        alpha_star = flatten_matrix(alpha_blade,num_cpt,num_mic,num_rot,num_blades,num_sec,num_cf)
        U          = flatten_matrix(U,num_cpt,num_mic,num_rot,num_blades,num_sec,num_cf)
        cos_zeta_r = flatten_matrix(cos_zeta_r,num_cpt,num_mic,num_rot,num_blades,num_sec,num_cf)
        M_tot      = flatten_matrix(M_tot,num_cpt,num_mic,num_rot,num_blades,num_sec,num_cf)
        f          = flatten_matrix(f,num_cpt,num_mic,num_rot,num_blades,num_sec,num_cf)
        c_0        = flatten_matrix(c_0,num_cpt,num_mic,num_rot,num_blades,num_sec,num_cf)
        rho        = flatten_matrix(rho,num_cpt,num_mic,num_rot,num_blades,num_sec,num_cf)
        r_er       = flatten_matrix(r_er,num_cpt,num_mic,num_rot,num_blades,num_sec,num_cf)
        mu         = flatten_matrix(mu,num_cpt,num_mic,num_rot,num_blades,num_sec,num_cf)
        L          = flatten_matrix(L,num_cpt,num_mic,num_rot,num_blades,num_sec,num_cf)
        M          = flatten_matrix(M,num_cpt,num_mic,num_rot,num_blades,num_sec,num_cf)
        Phi_er     = flatten_matrix(Phi_er,num_cpt,num_mic,num_rot,num_blades,num_sec,num_cf)
        Theta_er   = flatten_matrix(Theta_er,num_cpt,num_mic,num_rot,num_blades,num_sec,num_cf)
        
        
        '''calculation of boundary layer properties
        eqns 2 - 16 '''   
        
        # boundary layer properies of tripped and untripped at 0 angle of attack   
        boundary_layer_data  = compute_BPM_boundary_layer_properties(R_c,c,alpha_star) 
        
    
        ''' 
        define simulation variables/constants  
        '''        
        
        Re_delta_star_p_untripped = boundary_layer_data.delta_star_p_untripped*U*rho/mu
        Re_delta_star_p_tripped   = boundary_layer_data.delta_star_p_tripped*U*rho/mu 
        
    
        ''' 
        calculation of directivitiy terms 
        eqns 24 - 50
        '''   
        Dbar_h, Dbar_l = compute_noise_directivities(Theta_er,Phi_er,cos_zeta_r,M_tot)
        
        
        ''' 
        calculation of turbulent boundary layer - trailing edge noise 
        eqns 24 - 50
        '''  
            
        SPL_TBL_TE_tripped   = compute_TBL_TE_broadband_noise(f,r_er,L,U,M,R_c,Dbar_h,Dbar_l,Re_delta_star_p_tripped,
                                                  boundary_layer_data.delta_star_p_tripped,
                                                  boundary_layer_data.delta_star_s_tripped,
                                                  alpha_star) 
        
        SPL_TBL_TE_untripped = compute_TBL_TE_broadband_noise(f,r_er,L,U,M,R_c,Dbar_h,Dbar_l,Re_delta_star_p_untripped,
                                                  boundary_layer_data.delta_star_p_untripped,
                                                  boundary_layer_data.delta_star_s_untripped,
                                                  alpha_star)  
     
        '''
        calculation of laminar boundary layer - vortex shedding
        eqns 53 - 60
        '''
        SPL_LBL_VS = compute_LBL_VS_broadband_noise(R_c,alpha_star,boundary_layer_data.delta_star_p_untripped,r_er,L,M,Dbar_h,f,U)
      
    
        '''
        calculation of tip vortex noise
        eqns 61 - 67
        '''
        alpha_TIP = alpha_star
        SPL_TIP   = compute_TIP_broadband_noise(alpha_TIP,M,c,c_0,f,Dbar_h,r_er)  
        

        '''
        TO DO : Compute BWI
        '''
        

        '''
        TO DO : Compute BVI
        '''        
 
        SPL_TBL_TE_tripped   = unflatten_matrix(SPL_TBL_TE_tripped,num_cpt,num_mic,num_rot,num_blades,num_sec,num_cf) 
        SPL_TBL_TE_untripped = unflatten_matrix(SPL_TBL_TE_untripped,num_cpt,num_mic,num_rot,num_blades,num_sec,num_cf)
        SPL_LBL_VS           = unflatten_matrix(SPL_LBL_VS,num_cpt,num_mic,num_rot,num_blades,num_sec,num_cf)          
        SPL_TIP              = unflatten_matrix(SPL_TIP,num_cpt,num_mic,num_rot,num_blades,num_sec,num_cf)   
 
        broadband_SPLs    = np.zeros((3,num_cpt,num_mic,num_rot,num_blades,num_sec,num_cf))
        broadband_SPLs[0] = SPL_TBL_TE_tripped 
        broadband_SPLs[1] = SPL_LBL_VS
        broadband_SPLs[2] = SPL_TIP          
        SPL_rotor         = SPL_arithmetic(broadband_SPLs, sum_axis=0) 
         
        res.SPL_prop_broadband_spectrum                   = SPL_arithmetic(SPL_arithmetic(SPL_rotor,sum_axis=3),sum_axis=3)
        res.SPL_prop_broadband_spectrum_dBA               = A_weighting(res.SPL_prop_broadband_spectrum,frequency) 
        res.SPL_prop_broadband_1_3_spectrum               = res.SPL_prop_broadband_spectrum # already in 1/3 octave specturm 
        res.SPL_prop_broadband_1_3_spectrum_dBA           = res.SPL_prop_broadband_spectrum_dBA # already in 1/3 octave specturm 
        
    return
 
def flatten_matrix(x,num_cpt,num_mic,num_rot,num_blades,num_sec,num_cf):
    return np.reshape(x,(num_cpt*num_mic*num_rot*num_blades*num_sec*num_cf))


def unflatten_matrix(x,num_cpt,num_mic,num_rot,num_blades,num_sec,num_cf):
    return np.reshape(x,(num_cpt,num_mic,num_rot,num_blades,num_sec,num_cf))