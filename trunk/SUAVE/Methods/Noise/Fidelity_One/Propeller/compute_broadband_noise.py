## @ingroup Methods-Noise-Fidelity_One-Propeller
# compute_broadband_noise.py
#
# Created:   Mar 2021, M. Clarke
# Modified:  Feb 2022, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np 

from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.dbA_noise                     import A_weighting
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.SPL_harmonic_to_third_octave  import SPL_harmonic_to_third_octave  
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.decibel_arithmetic            import SPL_arithmetic
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_airfoil_properties  import compute_airfoil_properties
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_naca_4series import compute_naca_4series  
from scipy.special import fresnel
 
# ----------------------------------------------------------------------
# Frequency Domain Broadband Noise Computation
# ----------------------------------------------------------------------

## @ingroup Methods-Noise-Fidelity_One-Propeller   
def compute_broadband_noise(freestream,angle_of_attack,bspv,
                            velocity_vector,rotors,aeroacoustic_data,settings,res):
    '''This computes the trailing edge noise compoment of broadband noise of a propeller or 
    lift-rotor in the frequency domain. Boundary layer properties are computed using SUAVE's 
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
    num_rot       = len(bspv.blade_section_coordinate_sys[0,0,:,0,0,0,0,0])
    num_mic       = len(bspv.blade_section_coordinate_sys[0,:,0,0,0,0,0,0])  
    rotor         = rotors[list(rotors.keys())[0]]
    frequency     = settings.center_frequencies
    num_cf        = len(frequency)     
    
    # ----------------------------------------------------------------------------------
    # Trailing Edge Noise
    # ---------------------------------------------------------------------------------- 
    p_ref              = 2E-5                               # referece atmospheric pressure
    c_0                = freestream.speed_of_sound          # speed of sound
    rho                = freestream.density                 # air density 
    dyna_visc          = freestream.dynamic_viscosity
    kine_visc          = dyna_visc/rho                      # kinematic viscousity    
    alpha_blade        = aeroacoustic_data.disc_effective_angle_of_attack 
    Vt_2d              = aeroacoustic_data.disc_tangential_velocity  
    Va_2d              = aeroacoustic_data.disc_axial_velocity                
    blade_chords       = rotor.chord_distribution           # blade chord    
    r                  = rotor.radius_distribution          # radial location  
    a_geo              = rotor.airfoil_geometry
    num_sec            = len(r) 
    num_azi            = len(aeroacoustic_data.disc_effective_angle_of_attack[0,0,:])   
    
    if a_geo == None: # if not airfoil geometry defined, get surrogates for NACA 0012 and use to compute noise 
        print('No airfoil defined, NACA 0012 surrogates will be used for broadband noise model') 
        airfoil_geometry              = compute_naca_4series( ['0012'], npoints= 100)   
        a_loc                         = list(np.zeros(num_sec).astype(int)) 
        airfoil_data                  = compute_airfoil_properties(airfoil_geometry)  
        a_names                       = airfoil_data.airfoil_names                
        theta_lower_surface_surs      = airfoil_data.theta_lower_surface_surrogates       
        delta_lower_surface_surs      = airfoil_data.delta_lower_surface_surrogates     
        delta_star_lower_surface_surs = airfoil_data.delta_star_lower_surface_surrogates   
        ue_lower_surface_surs         = airfoil_data.ue_lower_surface_surrogates            
        cf_lower_surface_surs         = airfoil_data.cf_lower_surface_surrogates        
        dcp_dx_lower_surface_surs     = airfoil_data.dcp_dx_lower_surface_surrogates      
        H_lower_surface_surs          = airfoil_data.H_lower_surface_surrogates         
        theta_upper_surface_surs      = airfoil_data.theta_upper_surface_surrogates         
        delta_upper_surface_surs      = airfoil_data.delta_upper_surface_surrogates     
        delta_star_upper_surface_surs = airfoil_data.delta_star_upper_surface_surrogates         
        ue_upper_surface_surs         = airfoil_data.ue_upper_surface_surrogates            
        cf_upper_surface_surs         = airfoil_data.cf_upper_surface_surrogates        
        dcp_dx_upper_surface_surs     = airfoil_data.dcp_dx_upper_surface_surrogates      
        H_upper_surface_surs          = airfoil_data.H_upper_surface_surrogates  
        
    else: 
        airfoil_data                   = rotor.airfoil_data
        a_loc                          = rotor.airfoil_polar_stations  
        a_names                        = rotor.airfoil_geometry_files
        theta_lower_surface_surs       = airfoil_data.theta_lower_surface_surrogates       
        delta_lower_surface_surs       = airfoil_data.delta_lower_surface_surrogates     
        delta_star_lower_surface_surs  = airfoil_data.delta_star_lower_surface_surrogates   
        ue_lower_surface_surs          = airfoil_data.ue_lower_surface_surrogates            
        cf_lower_surface_surs          = airfoil_data.cf_lower_surface_surrogates        
        dcp_dx_lower_surface_surs      = airfoil_data.dcp_dx_lower_surface_surrogates      
        H_lower_surface_surs           = airfoil_data.H_lower_surface_surrogates         
        theta_upper_surface_surs       = airfoil_data.theta_upper_surface_surrogates         
        delta_upper_surface_surs       = airfoil_data.delta_upper_surface_surrogates     
        delta_star_upper_surface_surs  = airfoil_data.delta_star_upper_surface_surrogates         
        ue_upper_surface_surs          = airfoil_data.ue_upper_surface_surrogates            
        cf_upper_surface_surs          = airfoil_data.cf_upper_surface_surrogates        
        dcp_dx_upper_surface_surs      = airfoil_data.dcp_dx_upper_surface_surrogates      
        H_upper_surface_surs           = airfoil_data.H_upper_surface_surrogates         
 

    dim_sur            = len(theta_lower_surface_surs)    
    U_blade            = np.sqrt(Vt_2d**2 + Va_2d**2)
    Re_blade           = U_blade*np.repeat(np.repeat(blade_chords[np.newaxis,:],num_cpt,axis=0)[:,:,np.newaxis],num_azi,axis=2)*\
                          np.repeat(np.repeat((rho/dyna_visc),num_sec,axis=1)[:,:,np.newaxis],num_azi,axis=2)
    rho_blade          = np.repeat(np.repeat(rho,num_sec,axis=1)[:,:,np.newaxis],num_azi,axis=2)
    U_inf              = np.atleast_2d(np.linalg.norm(velocity_vector,axis=1)).T
    M                  = U_inf/c_0                                             
    B                  = rotor.number_of_blades             # number of rotor blades
    Omega              = aeroacoustic_data.omega            # angular velocity   
    beta_sq            = 1 - M**2                                  
    delta_r            = np.zeros_like(r)
    del_r              = r[1:] - r[:-1]
    delta_r[0]         = 2*del_r[0]
    delta_r[-1]        = 2*del_r[-1]
    delta_r[1:-1]      = (del_r[:-1]+ del_r[1:])/2 


    if np.all(Omega == 0):
        res.p_pref_broadband                          = np.zeros((num_cpt,num_mic,num_rot,num_cf)) 
        res.SPL_prop_broadband_spectrum               = np.zeros_like(res.p_pref_broadband)
        res.SPL_prop_broadband_spectrum_dBA           = np.zeros_like(res.p_pref_broadband)
        res.SPL_prop_broadband_1_3_spectrum           = np.zeros((num_cpt,num_mic,num_rot,num_cf))
        res.SPL_prop_broadband_1_3_spectrum_dBA       = np.zeros((num_cpt,num_mic,num_rot,num_cf))
        res.p_pref_azimuthal_broadband                = np.zeros((num_cpt,num_mic,num_rot,num_azi,num_cf))
        res.p_pref_azimuthal_broadband_dBA            = np.zeros_like(res.p_pref_azimuthal_broadband)
        res.SPL_prop_azimuthal_broadband_spectrum     = np.zeros_like(res.p_pref_azimuthal_broadband)
        res.SPL_prop_azimuthal_broadband_spectrum_dBA = np.zeros_like(res.p_pref_azimuthal_broadband)
    else:
        delta        = np.zeros((num_cpt,num_mic,num_rot,num_sec,num_azi,num_cf,2)) #  control points ,  number rotors, number blades , number sections , sides of airfoil
        delta_star   = np.zeros_like(delta)
        dp_dx        = np.zeros_like(delta)
        tau_w        = np.zeros_like(delta)
        Ue           = np.zeros_like(delta)
        Theta        = np.zeros_like(delta)

        lower_surface_theta      = np.zeros((num_sec,num_azi))
        lower_surface_delta      = np.zeros_like(lower_surface_theta)
        lower_surface_delta_star = np.zeros_like(lower_surface_theta)
        lower_surface_cf         = np.zeros_like(lower_surface_theta)
        lower_surface_Ue         = np.zeros_like(lower_surface_theta)
        lower_surface_H          = np.zeros_like(lower_surface_theta)
        lower_surface_dp_dx      = np.zeros_like(lower_surface_theta)
        upper_surface_theta      = np.zeros_like(lower_surface_theta)
        upper_surface_delta      = np.zeros_like(lower_surface_theta)
        upper_surface_delta_star = np.zeros_like(lower_surface_theta)
        upper_surface_cf         = np.zeros_like(lower_surface_theta)
        upper_surface_Ue         = np.zeros_like(lower_surface_theta)
        upper_surface_H          = np.zeros_like(lower_surface_theta)
        upper_surface_dp_dx      = np.zeros_like(lower_surface_theta)

        # ------------------------------------------------------------
        # ****** TRAILING EDGE BOUNDARY LAYER PROPERTY CALCULATIONS  ******   
        if rotor.nonuniform_freestream: 

            # return the 1D Cl and CDval of shape (ctrl_pts, Nr)
            theta_ls         = np.zeros((num_cpt,num_sec,num_azi))
            delta_ls         = np.zeros((num_cpt,num_sec,num_azi))
            delta_star_ls    = np.zeros((num_cpt,num_sec,num_azi))
            ue_ls            = np.zeros((num_cpt,num_sec,num_azi))
            cf_ls            = np.zeros((num_cpt,num_sec,num_azi))
            dcp_dx_ls        = np.zeros((num_cpt,num_sec,num_azi))
            H_ls             = np.zeros((num_cpt,num_sec,num_azi))
            theta_us         = np.zeros((num_cpt,num_sec,num_azi))
            delta_us         = np.zeros((num_cpt,num_sec,num_azi))
            delta_star_us    = np.zeros((num_cpt,num_sec,num_azi))
            ue_us            = np.zeros((num_cpt,num_sec,num_azi))
            cf_us            = np.zeros((num_cpt,num_sec,num_azi))
            dcp_dx_us        = np.zeros((num_cpt,num_sec,num_azi))
            H_us             = np.zeros((num_cpt,num_sec,num_azi))
            
            for i_azi in range(num_azi):  
                for jj in range(dim_sur):
                    local_aoa                     = alpha_blade[:,:,i_azi]
                    local_Re                      = Re_blade[:,:,i_azi]
                    
                    theta_ls_data                 = theta_lower_surface_surs[a_names[jj]]((local_Re,local_aoa))
                    delta_ls_data                 = delta_lower_surface_surs[a_names[jj]]((local_Re,local_aoa))
                    delta_star_ls_data            = delta_star_lower_surface_surs[a_names[jj]]((local_Re,local_aoa)) 
                    ue_ls_data                    = ue_lower_surface_surs[a_names[jj]]((local_Re,local_aoa))
                    cf_ls_data                    = cf_lower_surface_surs[a_names[jj]]((local_Re,local_aoa))
                    dcp_dx_ls_data                = dcp_dx_lower_surface_surs[a_names[jj]]((local_Re,local_aoa))
                    H_ls_data                     = H_lower_surface_surs[a_names[jj]]((local_Re,local_aoa))
                    theta_us_data                 = theta_upper_surface_surs[a_names[jj]]((local_Re,local_aoa))
                    delta_us_data                 = delta_upper_surface_surs[a_names[jj]]((local_Re,local_aoa))
                    delta_star_us_data            = delta_star_upper_surface_surs[a_names[jj]]((local_Re,local_aoa)) 
                    ue_us_data                    = ue_upper_surface_surs[a_names[jj]]((local_Re,local_aoa))
                    cf_us_data                    = cf_upper_surface_surs[a_names[jj]]((local_Re,local_aoa)) 
                    dcp_dx_us_data                = dcp_dx_upper_surface_surs[a_names[jj]]((local_Re,local_aoa))
                    H_us_data                     = H_upper_surface_surs[a_names[jj]]((local_Re,local_aoa)) 
                    locs                          = np.where(np.array(a_loc) == jj ) 
                    
                    theta_ls[:,locs,i_azi]        = theta_ls_data[:,locs]
                    delta_ls[:,locs,i_azi]        = delta_ls_data[:,locs]
                    delta_star_ls[:,locs,i_azi]   = delta_star_ls_data[:,locs]     
                    ue_ls[:,locs,i_azi]           = ue_ls_data[:,locs]       
                    cf_ls[:,locs,i_azi]           = cf_ls_data[:,locs]     
                    dcp_dx_ls[:,locs,i_azi]       = dcp_dx_ls_data[:,locs]
                    H_ls[:,locs,i_azi]            = H_ls_data[:,locs]         
                    theta_us[:,locs,i_azi]        = theta_us_data[:,locs]     
                    delta_us[:,locs,i_azi]        = delta_us_data[:,locs]     
                    delta_star_us[:,locs,i_azi]   = delta_star_us_data[:,locs]   
                    ue_us[:,locs,i_azi]           = ue_us_data[:,locs]        
                    cf_us[:,locs,i_azi]           = cf_us_data[:,locs]      
                    dcp_dx_us[:,locs,i_azi]       = dcp_dx_us_data[:,locs]   
                    H_us[:,locs,i_azi]            = H_us_data[:,locs]    
                
            blade_chords_3d           = np.tile(np.tile(blade_chords[None,:],(num_cpt,1))[:,:,None],(1,1,num_azi))
            dP_dX_ls                  = dcp_dx_ls*(0.5*rho_blade*U_blade**2)/blade_chords_3d
            dP_dX_us                  = dcp_dx_us*(0.5*rho_blade*U_blade**2)/blade_chords_3d
            
            lower_surface_theta       = theta_ls   
            lower_surface_delta       = delta_ls  
            lower_surface_delta_star  = delta_star_ls 
            lower_surface_cf          = cf_ls 
            lower_surface_Ue          = ue_ls 
            lower_surface_H           = H_ls 
            lower_surface_dp_dx       = dP_dX_ls 
            upper_surface_theta       = theta_us 
            upper_surface_delta       = delta_us 
            upper_surface_delta_star  = delta_star_us 
            upper_surface_cf          = cf_us 
            upper_surface_Ue          = ue_us 
            upper_surface_H           = H_us 
            upper_surface_dp_dx       = dP_dX_us  
                
        else: 
            theta_ls         = np.zeros((num_cpt,num_sec))
            delta_ls         = np.zeros((num_cpt,num_sec))
            delta_star_ls    = np.zeros((num_cpt,num_sec))
            ue_ls            = np.zeros((num_cpt,num_sec))
            cf_ls            = np.zeros((num_cpt,num_sec))
            dcp_dx_ls        = np.zeros((num_cpt,num_sec))
            H_ls             = np.zeros((num_cpt,num_sec))
            theta_us         = np.zeros((num_cpt,num_sec))
            delta_us         = np.zeros((num_cpt,num_sec))
            delta_star_us    = np.zeros((num_cpt,num_sec))
            ue_us            = np.zeros((num_cpt,num_sec))
            cf_us            = np.zeros((num_cpt,num_sec))
            dcp_dx_us        = np.zeros((num_cpt,num_sec))
            H_us             = np.zeros((num_cpt,num_sec))  
            local_aoa        = alpha_blade[:,:,0]
            local_Re         = Re_blade[:,:,0]            

            for jj in range(dim_sur):
                theta_ls_data           = theta_lower_surface_surs[a_names[jj]]((local_Re,local_aoa))
                delta_ls_data           = delta_lower_surface_surs[a_names[jj]]((local_Re,local_aoa))
                delta_star_ls_data      = delta_star_lower_surface_surs[a_names[jj]]((local_Re,local_aoa)) 
                ue_ls_data              = ue_lower_surface_surs[a_names[jj]]((local_Re,local_aoa))
                cf_ls_data              = cf_lower_surface_surs[a_names[jj]]((local_Re,local_aoa))
                dcp_dx_ls_data          = dcp_dx_lower_surface_surs[a_names[jj]]((local_Re,local_aoa))
                H_ls_data               = H_lower_surface_surs[a_names[jj]]((local_Re,local_aoa))
                theta_us_data           = theta_upper_surface_surs[a_names[jj]]((local_Re,local_aoa))
                delta_us_data           = delta_upper_surface_surs[a_names[jj]]((local_Re,local_aoa))
                delta_star_us_data      = delta_star_upper_surface_surs[a_names[jj]]((local_Re,local_aoa)) 
                ue_us_data              = ue_upper_surface_surs[a_names[jj]]((local_Re,local_aoa))
                cf_us_data              = cf_upper_surface_surs[a_names[jj]]((local_Re,local_aoa)) 
                dcp_dx_us_data          = dcp_dx_upper_surface_surs[a_names[jj]]((local_Re,local_aoa))
                H_us_data               = H_upper_surface_surs[a_names[jj]]((local_Re,local_aoa))
                
                locs                    = np.where(np.array(a_loc) == jj ) 
                
                theta_ls[:,locs]        = theta_ls_data[:,locs]
                delta_ls[:,locs]        = delta_ls_data[:,locs]
                delta_star_ls[:,locs]   = delta_star_ls_data[:,locs]     
                ue_ls[:,locs]           = ue_ls_data[:,locs]       
                cf_ls[:,locs]           = cf_ls_data[:,locs]     
                dcp_dx_ls[:,locs]       = dcp_dx_ls_data[:,locs]
                H_ls[:,locs]            = H_ls_data[:,locs]         
                theta_us[:,locs]        = theta_us_data[:,locs]     
                delta_us[:,locs]        = delta_us_data[:,locs]     
                delta_star_us[:,locs]   = delta_star_us_data[:,locs]   
                ue_us[:,locs]           = ue_us_data[:,locs]        
                cf_us[:,locs]           = cf_us_data[:,locs]      
                dcp_dx_us[:,locs]       = dcp_dx_us_data[:,locs]   
                H_us[:,locs]            = H_us_data[:,locs]    
            
            blade_chords_2d           = np.tile(blade_chords[None,:],(num_cpt,1))
            dP_dX_ls                  = dcp_dx_ls*(0.5*rho_blade[:,:,0]*U_blade[:,:,0]**2)/blade_chords_2d
            dP_dX_us                  = dcp_dx_us*(0.5*rho_blade[:,:,0]*U_blade[:,:,0]**2)/blade_chords_2d
            
            lower_surface_theta       = np.tile(theta_ls[:,:,None],(1,1,num_azi))  
            lower_surface_delta       = np.tile(delta_ls[:,:,None],(1,1,num_azi)) 
            lower_surface_delta_star  = np.tile(delta_star_ls[:,:,None],(1,1,num_azi)) 
            lower_surface_cf          = np.tile(cf_ls[:,:,None],(1,1,num_azi))  
            lower_surface_Ue          = np.tile(ue_ls[:,:,None],(1,1,num_azi))  
            lower_surface_H           = np.tile(H_ls[:,:,None],(1,1,num_azi))  
            lower_surface_dp_dx       = np.tile(dP_dX_ls[:,:,None],(1,1,num_azi)) 
            upper_surface_theta       = np.tile(theta_us[:,:,None],(1,1,num_azi))  
            upper_surface_delta       = np.tile(delta_us[:,:,None],(1,1,num_azi))  
            upper_surface_delta_star  = np.tile(delta_star_us[:,:,None],(1,1,num_azi)) 
            upper_surface_cf          = np.tile(cf_us[:,:,None],(1,1,num_azi))  
            upper_surface_Ue          = np.tile(ue_us[:,:,None],(1,1,num_azi))  
            upper_surface_H           = np.tile(H_us[:,:,None],(1,1,num_azi))  
            upper_surface_dp_dx       = np.tile(dP_dX_us[:,:,None],(1,1,num_azi))  
            
        # replace nans 0 with mean as a post post-processor
        lower_surface_theta       = np.nan_to_num(lower_surface_theta)
        upper_surface_theta       = np.nan_to_num(upper_surface_theta)
        lower_surface_delta       = np.nan_to_num(lower_surface_delta)
        upper_surface_delta       = np.nan_to_num(upper_surface_delta)
        lower_surface_delta_star  = np.nan_to_num(lower_surface_delta_star)
        upper_surface_delta_star  = np.nan_to_num(upper_surface_delta_star)
        lower_surface_cf          = np.nan_to_num(lower_surface_cf)
        upper_surface_cf          = np.nan_to_num(upper_surface_cf)
        lower_surface_dp_dx       = np.nan_to_num(lower_surface_dp_dx )
        upper_surface_dp_dx       = np.nan_to_num(upper_surface_dp_dx )
        lower_surface_Ue          = np.nan_to_num(lower_surface_Ue)
        upper_surface_Ue          = np.nan_to_num(upper_surface_Ue)
        lower_surface_H           = np.nan_to_num(lower_surface_H)
        upper_surface_H           = np.nan_to_num(upper_surface_H)
        
        # apply thresholds for non-converged boundary layer solutions form pandel code 
        lower_surface_theta[abs(lower_surface_theta)> 0.01 ]           = 0.0
        upper_surface_theta[abs(upper_surface_theta)>0.01 ]            = 0.0
        lower_surface_delta[abs(lower_surface_delta)> 0.1 ]            = 0.0
        upper_surface_delta[abs(upper_surface_delta)> 0.1]             = 0.0
        lower_surface_delta_star[abs(lower_surface_delta_star)>0.1 ]   = 0.0
        upper_surface_delta_star[abs(upper_surface_delta_star)>0.1 ]   = 0.0
        lower_surface_cf[abs(lower_surface_cf)>0.1 ]                   = 0.0
        upper_surface_cf[abs(upper_surface_cf)> 0.1]                   = 0.0
        lower_surface_dp_dx[abs(lower_surface_dp_dx)> 1E7]             = 0.0
        upper_surface_dp_dx[abs(upper_surface_dp_dx)> 1E7]             = 0.0
        lower_surface_Ue[abs(lower_surface_Ue)> 500.]                  = 0.0
        upper_surface_Ue[abs(upper_surface_Ue)> 500.]                  = 0.0
        lower_surface_H[abs(lower_surface_H)> 10.]                     = 0.0
        upper_surface_H[abs(upper_surface_H)> 10.]                     = 0.0 
        
        # replace null solutions with mean
        lower_surface_theta[lower_surface_theta == 0]                  = np.mean(lower_surface_theta)
        upper_surface_theta[upper_surface_theta == 0]                  = np.mean(upper_surface_theta)
        lower_surface_delta[lower_surface_delta == 0]                  = np.mean(lower_surface_delta)
        upper_surface_delta[upper_surface_delta == 0]                  = np.mean(upper_surface_delta)
        lower_surface_delta_star[lower_surface_delta_star == 0]        = np.mean(lower_surface_delta_star)
        upper_surface_delta_star[upper_surface_delta_star== 0]         = np.mean(upper_surface_delta_star)
        lower_surface_cf[lower_surface_cf == 0]                        = np.mean(lower_surface_cf)
        upper_surface_cf[upper_surface_cf == 0]                        = np.mean(upper_surface_cf)
        lower_surface_dp_dx [lower_surface_dp_dx  == 0]                = np.mean(lower_surface_dp_dx )
        upper_surface_dp_dx [upper_surface_dp_dx  == 0]                = np.mean(upper_surface_dp_dx )
        lower_surface_Ue[lower_surface_Ue == 0]                        = np.mean(lower_surface_Ue)
        upper_surface_Ue[upper_surface_Ue == 0]                        = np.mean(upper_surface_Ue)
        lower_surface_H[lower_surface_H == 0]                          = np.mean(lower_surface_H)
        upper_surface_H[upper_surface_H == 0]                          = np.mean(upper_surface_H)

        # ------------------------------------------------------------
        # ****** TRAILING EDGE BOUNDARY LAYER PROPERTY CALCULATIONS  ******

        delta[:,:,:,:,:,:,0]        = np.tile(lower_surface_delta[:,None,None,:,:,None],(1,num_mic,num_rot,1,1,num_cf))      # lower surface boundary layer thickness
        delta[:,:,:,:,:,:,1]        = np.tile(upper_surface_delta[:,None,None,:,:,None],(1,num_mic,num_rot,1,1,num_cf))      # upper surface boundary layer thickness
        delta_star[:,:,:,:,:,:,0]   = np.tile(lower_surface_delta_star[:,None,None,:,:,None],(1,num_mic,num_rot,1,1,num_cf)) # lower surface displacement thickness
        delta_star[:,:,:,:,:,:,1]   = np.tile(upper_surface_delta_star[:,None,None,:,:,None],(1,num_mic,num_rot,1,1,num_cf)) # upper surface displacement thickness
        dp_dx[:,:,:,:,:,:,0]        = np.tile(lower_surface_dp_dx[:,None,None,:,:,None],(1,num_mic,num_rot,1,1,num_cf))      # lower surface pressure differential
        dp_dx[:,:,:,:,:,:,1]        = np.tile(upper_surface_dp_dx[:,None,None,:,:,None],(1,num_mic,num_rot,1,1,num_cf))      # upper surface pressure differential
        Ue[:,:,:,:,:,:,0]           = np.tile(lower_surface_Ue[:,None,None,:,:,None],(1,num_mic,num_rot,1,1,num_cf))         # lower surface boundary layer edge velocity
        Ue[:,:,:,:,:,:,1]           = np.tile(upper_surface_Ue[:,None,None,:,:,None],(1,num_mic,num_rot,1,1,num_cf))         # upper surface boundary layer edge velocity
        tau_w[:,:,:,:,:,:,0]        = np.tile((lower_surface_cf*(0.5*rho_blade*(U_blade**2)))[:,None,None,:,:,None],(1,num_mic,num_rot,1,1,num_cf))       # lower surface wall shear stress
        tau_w[::,:,:,:,:,:,1]       = np.tile((upper_surface_cf*(0.5*rho_blade*(U_blade**2)))[:,None,None,:,:,None],(1,num_mic,num_rot,1,1,num_cf))      # upper surface wall shear stress
        Theta[:,:,:,:,:,:,0]        = np.tile(lower_surface_theta[:,None,None,:,:,None],(1,num_mic,num_rot,1,1,num_cf))      # lower surface momentum thickness
        Theta[:,:,:,:,:,:,1]        = np.tile(upper_surface_theta[:,None,None,:,:,None],(1,num_mic,num_rot,1,1,num_cf))      # upper surface momentum thickness

        # Update dimensions for computation
        r         = np.tile(r[None,None,None,:,None,None],(num_cpt,num_mic,num_rot,1,num_azi,num_cf))
        c         = np.tile(blade_chords[None,None,None,:,None,None],(num_cpt,num_mic,num_rot,1,num_azi,num_cf))
        delta_r   = np.tile(delta_r[None,None,None,:,None,None],(num_cpt,num_mic,num_rot,1,num_azi,num_cf))
        M         = np.tile(M[:,None,None,None,None,None,:],(1,num_mic,num_rot,num_sec,num_azi,num_cf,1))  
        c_0       = np.tile(c_0[:,None,None,None,None,None,:],(1,num_mic,num_rot,num_sec,num_azi,num_cf,2))
        beta_sq   = np.tile(beta_sq[:,None,None,None,None,None,:],(1,num_mic,num_rot,num_sec,num_azi,num_cf,2))
        Omega     = np.tile(Omega[:,None,None,None,None,None,:],(1,num_mic,num_rot,num_sec,num_azi,num_cf,2))
        U_inf     = np.tile(U_inf[:,None,None,None,None,None,:],(1,num_mic,num_rot,num_sec,num_azi,num_cf,2))
        rho       = np.tile(rho[:,None,None,None,None,None,:],(1,num_mic,num_rot,num_sec,num_azi,num_cf,2))
        kine_visc = np.tile(kine_visc[:,None,None,None,None,None,:],(1,num_mic,num_rot,num_sec,num_azi,num_cf,2))

        X   = np.repeat(bspv.blade_section_coordinate_sys[:,:,:,:,:,:,0,:],2,axis = 6)
        Y   = np.repeat(bspv.blade_section_coordinate_sys[:,:,:,:,:,:,1,:],2,axis = 6)
        Z   = np.repeat(bspv.blade_section_coordinate_sys[:,:,:,:,:,:,2,:],2,axis = 6)

        # ------------------------------------------------------------
        # ****** BLADE MOTION CALCULATIONS ******
        # the rotational Mach number of the blade section
        omega   = np.tile((2*np.pi*frequency)[None,None,None,None,None,:,None],(num_cpt,num_mic,num_rot,num_sec,num_azi,1,2))
        r       = np.repeat(r[:,:,:,:,:,:,np.newaxis],2,axis = 6)
        c       = np.repeat(c[:,:,:,:,:,:,np.newaxis],2,axis = 6)/2
        delta_r = np.repeat(delta_r[:,:,:,:,:,:,np.newaxis],2,axis = 6)
        M       = np.repeat(M,2,axis = 6)
        R_s     = np.repeat(np.linalg.norm(bspv.blade_section_coordinate_sys,axis = 6),2,axis = 6)
        mu      = (omega/(1 +(Omega*r/c_0)*(X/R_s)))*M/(U_inf*beta_sq)

        # ------------------------------------------------------------
        # ****** LOADING TERM CALCULATIONS ******
        # equation 7
        epsilon       = X**2 + (beta_sq)*(Y**2 + Z**2)
        gamma         = np.sqrt(((mu/epsilon)**2)*(X**2 + beta_sq*(Z**2)))
        ss_1, cc_1    = fresnel(2*((((omega/(1 +  (Omega*r/c_0)*(X/R_s))) /(0.8*U_inf)) /c) + (mu/c)*M + (gamma/c)))
        ss_2, cc_2    = fresnel(2*((mu/c)*X/epsilon + (gamma/c)) )
        triangle      = (omega/(U_inf*c)) - (mu/c)*X/epsilon + (mu/c)*M
        norm_L_sq     = (1/triangle)*abs(np.exp(1j*2*triangle)*((1 - (1 + 1j)*(cc_1 - 1j*ss_1)) \
                        + ((np.exp(-1j*2*triangle))*(np.sqrt((((omega/(1 +  (Omega*r/c_0)*(X/R_s))) /(0.8*U_inf)) + mu*M + gamma)/(mu*X/epsilon +gamma))) \
                           *(1 + 1j)*(cc_2 - 1j*ss_2)) ))

        # ------------------------------------------------------------
        # ****** EMPIRICAL WALL PRESSURE SPECTRUM ******
        ones                     = np.ones_like(Theta)
        beta_c                   = (Theta/tau_w)*dp_dx 
        d                        = 4.76*((1.4/(delta/delta_star))**0.75)*(0.375*(3.7 + 1.5*beta_c) - 1)
        a                        = (2.82*((delta/delta_star)**2)*(np.power((6.13*((delta/delta_star)**(-0.75)) + d),(3.7 + 1.5*beta_c))))*\
                                   (4.2*((0.8*((beta_c + 0.5)**3/4))/(delta/delta_star)) + 1)
        d_star                   = d
        d_star[beta_c<0.5]       = np.maximum(ones,1.5*d)[beta_c<0.5]
        Phi_pp_expression        =  (np.maximum(a, (0.25*beta_c - 0.52)*a)*((omega*delta_star/Ue)**2))/(((4.76*((omega*delta_star/Ue)**0.75) \
                                    + d_star)**(3.7 + 1.5*beta_c))+ (np.power((8.8*(((delta/Ue)/(kine_visc/(((tau_w/rho)**0.5)**2)))**(-0.57))\
                                    *(omega*delta_star/Ue)),(np.minimum(3*ones,(0.139 + 3.1043*beta_c)) + 7)) ))
        Phi_pp                   = ((tau_w**2)*delta_star*Phi_pp_expression)/Ue
        Phi_pp[np.isinf(Phi_pp)] = 0.
        Phi_pp[np.isnan(Phi_pp)] = 0.
 
        # Power Spectral Density from each blade
        mult       = ((omega/c_0)**2)*(c**2)*delta_r*(1/(32*np.pi**2))*(B/(2*np.pi))
        int_x      = np.linspace(0,2*np.pi,num_azi)  
        S_pp       = mult[:,:,:,:,0,:,:]*np.trapz(((Z/(X**2 + (1-M**2)*(Y**2 + Z**2)))**2)*norm_L_sq*\
                                                  (1.6*(0.8*U_inf)/omega)*Phi_pp,x = int_x,axis = 4) 
            
        # Sound Pressure Level
        SPL                        = 10*np.log10((2*np.pi*abs(S_pp))/((p_ref)**2)) 
        SPL[np.isinf(SPL)]         = 0   
        SPL_rotor                  = SPL_arithmetic(SPL_arithmetic(SPL, sum_axis = 5 ), sum_axis = 3 )
        
        # convert to 1/3 octave spectrum
        f = np.repeat(np.atleast_2d(frequency),num_cpt,axis = 0)
 
        res.SPL_prop_broadband_spectrum                   = SPL_rotor
        res.SPL_prop_broadband_spectrum_dBA               = A_weighting(SPL_rotor,frequency) 
        res.SPL_prop_broadband_1_3_spectrum               = SPL_harmonic_to_third_octave(SPL_rotor,f,settings)
        res.SPL_prop_broadband_1_3_spectrum_dBA           = SPL_harmonic_to_third_octave(A_weighting(SPL_rotor,frequency),f,settings) 
        res.SPL_prop_broadband_1_3_spectrum               = SPL_harmonic_to_third_octave(SPL_rotor,f,settings)
        
    return

