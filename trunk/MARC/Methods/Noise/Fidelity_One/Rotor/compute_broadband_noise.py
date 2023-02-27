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
def compute_broadband_noise(freestream,angle_of_attack,bspv,
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

    num_cpt       = len(angle_of_attack)
    num_rot       = len(bspv.blade_section_coordinate_sys[0,0,:,0,0,0,0])
    num_mic       = len(bspv.blade_section_coordinate_sys[0,:,0,0,0,0,0])  
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
    velocity           = freestream.velocity
    kine_visc          = dyna_visc/density                  # kinematic viscousity    
    alpha              = aeroacoustic_data.blade_effective_angle_of_attack 
    X_e                = bspv.blade_section_coordinate_sys   
    Vt                 = aeroacoustic_data.blade_tangential_velocity  
    Va                 = aeroacoustic_data.blade_axial_velocity                
    blade_chords       = rotor.chord_distribution           # blade chord    
    r                  = rotor.radius_distribution          # radial location 
    num_sec            = len(r)                                 
    B                  = rotor.number_of_blades             # number of rotor blades
    Omega              = aeroacoustic_data.omega            # angular velocity    
    L                  = np.zeros_like(r)
    del_r              = r[1:] - r[:-1]
    L[0]               = 2*del_r[0]
    L[-1]              = 2*del_r[-1]
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
    
        c              = np.tile(blade_chords[None,None,None,:,None],(num_cpt,num_mic,num_rot,1,num_cf))
        f              = np.tile(frequency[None,None,None,None,:],(num_cpt,num_mic,num_rot,num_sec,1))
        alpha_blade    = np.tile(alpha[:,None,None,:,None],(1,num_mic,num_rot,1,num_cf))
        V              = np.sqrt(Vt**2 + Va**2)
        V_tot          = np.tile(V[:,None,None,:,None],(1,num_mic,num_rot,1,num_cf))
        U_inf          = np.tile(velocity[:,:,None,None,None],(1,num_mic,num_rot,num_sec,num_cf))
        nu             = np.tile(kine_visc[:,:,None,None,None],(1,num_mic,num_rot,num_sec,num_cf))
        c_0            = np.tile(speed_of_sound[:,:,None,None,None],(1,num_mic,num_rot,num_sec,num_cf))
        rho            = np.tile(density[:,:,None,None,None],(1,num_mic,num_rot,num_sec,num_cf))
        X_er           = np.zeros((num_cpt,num_mic,num_rot,num_sec,num_cf,3,1))
        R_c            = # V_tot*c/nu
        M_tot          =  # V_tot/c_0                   
        M_x            = U_inf/c_0  
        x_e            = X_e[:,:,:,:,0,0]
        y_e            = X_e[:,:,:,:,1,0]
        z_e            = X_e[:,:,:,:,2,0] 
        r_e            = np.sqrt(x_e**2 + y_e**2 + z_e**2)              
        theta_e        = np.arccos(x_e/r_e)       
        Theta_er       = np.arccos(np.cos(theta_e)*np.sqrt(1 - (M_x**2)*(np.sin(theta_e))**2) + M_x*(np.sin(theta_e))**2 )  
        Y_e            = np.sqrt(y_e**2 + z_e**2)  
        r_er           = Y_e/(np.sin(Theta_er))  
        
        x_er           = np.cos(Theta_er)*r_er                                        # eqn 21 , Brooks & Burley   
        y_er, z_er     = y_e,z_e
            
        # in the following expressions, we are assuming a compact blade
        X_er[:,:,:,:,:,0,0] = x_er
        X_er[:,:,:,:,:,1,0] = y_er
        X_er[:,:,:,:,:,2,0] = z_er
        zeta_r            = np.acos(np.dot(X_er, (V_tot/((r_er *V_tot))) ))              # eqn 18 , Brooks & Burley  
        sigma_sq          = (r_er**2)*((1 - M_tot*np.cos(zeta_r))**2)                   # eqn 16 , Brooks & Burley 
        Phi_er            = np.acos(y_er/ np.sqrt(y_er**2 + z_er**2))                   # eqn 21 , Brooks & Burley 
       
        # flatten 
        R_c        = 0
        c          = 0
        alpha_star = np.reshape(alpha_blade,(num_cpt*num_mic*num_rot*num_sec*num_cf))
        U          = np.reshape(U,(num_cpt*num_mic*num_rot*num_sec*num_cf)) 
        zeta_r     = 0
        M_tot      = 0
        f          = 0
        r_e        = 0
        L          = 0
        U          = 0
        M          = 0
        Dbar_h     = 0
        Dbar_l     = 0
          
        
        '''calculation of boundary layer properties
        eqns 2 - 16 '''   
        
        # boundary layer properies of tripped and untripped at 0 angle of attack   
        boundary_layer_data  = compute_BPM_boundary_layer_properties(R_c,c,alpha_star) 
        
    
        ''' 
        define simulation variables/constants  
        '''        
        
        R_delta_star_p_untripped = boundary_layer_data.delta_star_p_untripped*U*dyna_visc/rho 
        R_delta_star_p_tripped   = boundary_layer_data.delta_star_p_tripped*U*dyna_visc/rho 
        
    
        ''' 
        calculation of directivitiy terms 
        eqns 24 - 50
        '''   
        Dbar_h, Dbar_l = compute_noise_directivities(Theta_er,Phi_er,zeta_r,M_tot)
        
        
        ''' 
        calculation of turbulent boundary layer - trailing edge noise 
        eqns 24 - 50
        '''  
            
        SPL_TBL_TE_tripped   = compute_TBL_TE_broadband_noise(f,r_e,L,U,M,R_c,Dbar_h,Dbar_l,R_delta_star_p_tripped,
                                                  boundary_layer_data.delta_star_p_tripped,
                                                  boundary_layer_data.delta_star_s_tripped,
                                                  alpha_star) 
        
        SPL_TBL_TE_untripped = compute_TBL_TE_broadband_noise(f,r_e,L,U,M,R_c,Dbar_h,Dbar_l,R_delta_star_p_untripped,
                                                  boundary_layer_data.delta_star_p_untripped,
                                                  boundary_layer_data.delta_star_s_untripped,
                                                  alpha_star)  
     
        '''
        calculation of laminar boundary layer - vortex shedding
        eqns 53 - 60
        '''
        SPL_LBL_VS = compute_LBL_VS_broadband_noise(R_c,alpha_star,boundary_layer_data.delta_star_p_untripped,r_e,L,M,Dbar_h,f,U)
      
    
        '''
        calculation of tip vortex noise
        eqns 61 - 67
        '''
        alpha_TIP = alpha_star
        SPL_TIP   = compute_TIP_broadband_noise(alpha_TIP,M,c,c_0,f,Dbar_h,r_e)  
     
        res.SPL_prop_broadband_spectrum                   = SPL_rotor
        res.SPL_prop_broadband_spectrum_dBA               = A_weighting(SPL_rotor,frequency) 
        res.SPL_prop_broadband_1_3_spectrum               = convert_to_third_octave_band(SPL_rotor,f,settings)
        res.SPL_prop_broadband_1_3_spectrum_dBA           = convert_to_third_octave_band(A_weighting(SPL_rotor,frequency),f,settings)  
        
    return
 