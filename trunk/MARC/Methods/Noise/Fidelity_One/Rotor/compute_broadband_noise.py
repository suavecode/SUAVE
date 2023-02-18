## @ingroup Methods-Noise-Fidelity_One-Propeller
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
from MARC.Core.Utilities                                                       import interp2d
from MARC.Methods.Noise.Fidelity_One.Noise_Tools.dbA_noise                     import A_weighting
from MARC.Methods.Noise.Fidelity_One.Noise_Tools.convert_to_third_octave_band  import convert_to_third_octave_band  
from MARC.Methods.Noise.Fidelity_One.Noise_Tools.decibel_arithmetic            import SPL_arithmetic
from scipy.special                                                              import fresnel
 
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
    airfoils           = rotor.Airfoils
    a_loc              = rotor.airfoil_polar_stations 
    num_sec            = len(r) 
    num_azi            = len(aeroacoustic_data.disc_effective_angle_of_attack[0,0,:])    
    U_blade            = np.sqrt(Vt_2d**2 + Va_2d**2)
    Re_blade           = U_blade*np.repeat(np.repeat(blade_chords[np.newaxis,:],num_cpt,axis=0)[:,:,np.newaxis],num_azi,axis=2)/\
                          np.repeat(np.repeat((kine_visc),num_sec,axis=1)[:,:,np.newaxis],num_azi,axis=2)
    rho_blade          = np.repeat(np.repeat(rho,num_sec,axis=1)[:,:,np.newaxis],num_azi,axis=2)
    U                  = np.atleast_2d(np.linalg.norm(velocity_vector,axis=1)).T
    M                  = U/c_0                                             
    B                  = rotor.number_of_blades             # number of rotor blades
    Omega              = aeroacoustic_data.omega            # angular velocity    
    L                  = np.zeros_like(r)
    del_r              = r[1:] - r[:-1]
    L[0]               = 2*del_r[0]
    L[-1]              = 2*del_r[-1]
    L[1:-1]            = (del_r[:-1]+ del_r[1:])/2


    bstei   = 4      # bottom surface trailing edge index 
    ustei   = -bstei # upper surface trailing edge index 

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
    
        # return the 1D Cl and CDval of shape (ctrl_pts, Nr)
        lower_surface_theta        = np.zeros((num_cpt,num_sec,num_azi))
        lower_surface_delta        = np.zeros_like(lower_surface_theta)
        lower_surface_delta_star   = np.zeros_like(lower_surface_theta)
        lower_surface_Ue           = np.zeros_like(lower_surface_theta)
        lower_surface_cf           = np.zeros_like(lower_surface_theta)
        lower_surface_dcp_dx       = np.zeros_like(lower_surface_theta)
        upper_surface_theta        = np.zeros_like(lower_surface_theta)
        upper_surface_delta        = np.zeros_like(lower_surface_theta)
        upper_surface_delta_star   = np.zeros_like(lower_surface_theta)
        upper_surface_Ue           = np.zeros_like(lower_surface_theta)
        upper_surface_cf           = np.zeros_like(lower_surface_theta)
        upper_surface_dcp_dx       = np.zeros_like(lower_surface_theta) 
    
        aloc  = np.atleast_3d(np.array(a_loc))
        aloc  = np.broadcast_to(aloc,np.shape(lower_surface_theta))
    
        for jj,airfoil in enumerate(airfoils):  
            bl                            = airfoil.polars.boundary_layer
            theta_ls_data                 = interp2d(Re_blade,alpha_blade,bl.reynolds_numbers, bl.angle_of_attacks, bl.theta[:,:,bstei])     
            delta_ls_data                 = interp2d(Re_blade,alpha_blade,bl.reynolds_numbers, bl.angle_of_attacks, bl.delta[:,:,bstei])        
            delta_star_ls_data            = interp2d(Re_blade,alpha_blade,bl.reynolds_numbers, bl.angle_of_attacks, bl.delta_star[:,:,bstei])   
            Ue_Vinf_ls_data               = interp2d(Re_blade,alpha_blade,bl.reynolds_numbers, bl.angle_of_attacks, bl.Ue_Vinf[:,:,bstei])      
            cf_ls_data                    = interp2d(Re_blade,alpha_blade,bl.reynolds_numbers, bl.angle_of_attacks, bl.cf[:,:,bstei])           
            dcp_dx_ls_data                = interp2d(Re_blade,alpha_blade,bl.reynolds_numbers, bl.angle_of_attacks, bl.dcp_dx[:,:,bstei])       
            theta_us_data                 = interp2d(Re_blade,alpha_blade,bl.reynolds_numbers, bl.angle_of_attacks, bl.theta[:,:,ustei])        
            delta_us_data                 = interp2d(Re_blade,alpha_blade,bl.reynolds_numbers, bl.angle_of_attacks, bl.delta[:,:,ustei])       
            delta_star_us_data            = interp2d(Re_blade,alpha_blade,bl.reynolds_numbers, bl.angle_of_attacks, bl.delta_star[:,:,ustei])   
            Ue_Vinf_us_data               = interp2d(Re_blade,alpha_blade,bl.reynolds_numbers, bl.angle_of_attacks, bl.Ue_Vinf[:,:,ustei])   
            cf_us_data                    = interp2d(Re_blade,alpha_blade,bl.reynolds_numbers, bl.angle_of_attacks, bl.cf[:,:,ustei])          
            dcp_dx_us_data                = interp2d(Re_blade,alpha_blade,bl.reynolds_numbers, bl.angle_of_attacks, bl.dcp_dx[:,:,ustei])      
    
            locs                                 = np.where(np.array(a_loc) == jj )  
            lower_surface_theta[:,locs,:]        = theta_ls_data[:,locs]
            lower_surface_delta[:,locs,:]        = delta_ls_data[:,locs]         
            lower_surface_delta_star[:,locs,:]   = delta_star_ls_data[:,locs]    
            lower_surface_Ue[:,locs,:]           = Ue_Vinf_ls_data[:,locs]            
            lower_surface_cf[:,locs,:]           = cf_ls_data[:,locs]            
            lower_surface_dcp_dx[:,locs,:]       = dcp_dx_ls_data[:,locs]        
            upper_surface_theta[:,locs,:]        = theta_us_data[:,locs]
            upper_surface_delta[:,locs,:]        = delta_us_data[:,locs]         
            upper_surface_delta_star[:,locs,:]   = delta_star_us_data[:,locs]    
            upper_surface_Ue[:,locs,:]           = Ue_Vinf_us_data[:,locs]
            upper_surface_cf[:,locs,:]           = cf_us_data[:,locs]    
            upper_surface_dcp_dx[:,locs,:]       = dcp_dx_us_data[:,locs] 
    
        blade_chords_3d           = np.tile(np.tile(blade_chords[None,:],(num_cpt,1))[:,:,None],(1,1,num_azi))
        dP_dX_ls                  = lower_surface_dcp_dx*(0.5*rho_blade*U_blade**2)/blade_chords_3d
        dP_dX_us                  = upper_surface_dcp_dx*(0.5*rho_blade*U_blade**2)/blade_chords_3d
        
        lower_surface_delta       = lower_surface_delta*blade_chords_3d 
        upper_surface_delta       = upper_surface_delta*blade_chords_3d 
        lower_surface_Ue          = lower_surface_Ue*U_blade  
        upper_surface_Ue          = upper_surface_Ue*U_blade 
        lower_surface_dp_dx       = dP_dX_ls
        upper_surface_dp_dx       = dP_dX_us 
        
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
        tau_w[:,:,:,:,:,:,1]        = np.tile((upper_surface_cf*(0.5*rho_blade*(U_blade**2)))[:,None,None,:,:,None],(1,num_mic,num_rot,1,1,num_cf))      # upper surface wall shear stress
        Theta[:,:,:,:,:,:,0]        = np.tile(lower_surface_theta[:,None,None,:,:,None],(1,num_mic,num_rot,1,1,num_cf))      # lower surface momentum thickness
        Theta[:,:,:,:,:,:,1]        = np.tile(upper_surface_theta[:,None,None,:,:,None],(1,num_mic,num_rot,1,1,num_cf))      # upper surface momentum thickness
 
   
       # BPM model 
       Re_c           = 0
       Re_delta_star_p= 0 # Reynolds number based on momentum thickness 
       D_bar_h        = 0 # high frequency directivity function 
       
       # Turbulent Boundary Layer - Trailing Edge 
       #G_TBL_TE_p = ((delta_star[:,:,:,:,:,:,0]*(M**5) *L*D_bar_h)/(r_e**2))*H_p((f*delta_star[:,:,:,:,:,:,0]/U),M,Re_c.Re_delta_star_p)  # eqn 3
       #G_TBL_TE_s = ((delta_star[:,:,:,:,:,:,1]*(M**5) *L*D_bar_h)/(r_e**2))*H_p((f*delta_star[:,:,:,:,:,:,1]/U),M,Re_c)# eqn 4
       #G_TBL_TE_a = 0  # eqn 5
       
       #G_TBL_TE = G_TBL_TE_p + G_TBL_TE_s + G_TBL_TE_a  # eqn 2
       
       ## Laminar Boundary Layer - Vortex Shedding 
       #G_LBL_VS  = 0 # eqn 9
       
       ## Blunt Trailing Edge 
       
       #G_BTE   =0  # eqn 10 
       
       
       ## Tip Noise 
       #G_Tip = 0 # eqn 11 
       
       ## BWI
       #G_BWI = 0 
       
       ## Total Self Noise 
       #G_self   = G_TBL_TE + G_LBL_VS + G_BTE + G_Tip   # eqn 1 
   
   
   
   
        # Sound Pressure Level
        SPL                        = 10*np.log10((2*np.pi*abs(S_pp))/((p_ref)**2)) 
        SPL[np.isinf(SPL)]         = 0   
        SPL_rotor                  = SPL_arithmetic(SPL_arithmetic(SPL, sum_axis = 5 ), sum_axis = 3 )
        
        # convert to 1/3 octave spectrum
        f = np.repeat(np.atleast_2d(frequency),num_cpt,axis = 0)
 
        res.SPL_prop_broadband_spectrum                   = SPL_rotor
        res.SPL_prop_broadband_spectrum_dBA               = A_weighting(SPL_rotor,frequency) 
        res.SPL_prop_broadband_1_3_spectrum               = convert_to_third_octave_band(SPL_rotor,f,settings)
        res.SPL_prop_broadband_1_3_spectrum_dBA           = convert_to_third_octave_band(A_weighting(SPL_rotor,frequency),f,settings)  
        
    return
 
 def H_p():
     
     return 
 
 def H_s():
     
     return  
 
 def H_a_star():
     
     return   