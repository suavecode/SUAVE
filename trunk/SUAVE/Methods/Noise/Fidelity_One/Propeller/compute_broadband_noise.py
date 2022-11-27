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
from SUAVE.Core import to_jnumpy
from jax import  jit
import jax.numpy as jnp
from jax.numpy import where as w
from jax.numpy import newaxis as na
#from scipy.special import fresnel
from tensorflow.python.ops.special_math_ops import fresnel_sin, fresnel_cos
from jax.experimental import jax2tf
from SUAVE.Core.Utilities                                                       import interp2d
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.dbA_noise                     import A_weighting
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.SPL_harmonic_to_third_octave  import SPL_harmonic_to_third_octave  
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.decibel_arithmetic            import SPL_arithmetic 
 
# ----------------------------------------------------------------------
# Frequency Domain Broadband Noise Computation
# ----------------------------------------------------------------------
## @ingroup Methods-Noise-Fidelity_One-Propeller 
@jit
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
    c_0                = to_jnumpy(freestream.speed_of_sound)         # speed of sound
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
    
    U_blade            = to_jnumpy(np.sqrt(Vt_2d**2 + Va_2d**2))
    Re_blade           = to_jnumpy(U_blade*np.repeat(np.repeat(blade_chords[np.newaxis,:],num_cpt,axis=0)[:,:,np.newaxis],num_azi,axis=2)/\
                          np.repeat(np.repeat((kine_visc),num_sec,axis=1)[:,:,np.newaxis],num_azi,axis=2))
    rho_blade          = to_jnumpy(np.repeat(np.repeat(rho,num_sec,axis=1)[:,:,np.newaxis],num_azi,axis=2))
    U_inf              = to_jnumpy(np.atleast_2d(np.linalg.norm(velocity_vector,axis=1)).T)
    M                  = U_inf/c_0                                             
    B                  = rotor.number_of_blades             # number of rotor blades
    Omega              = to_jnumpy(aeroacoustic_data.omega)            # angular velocity   
    beta_sq            = 1 - M**2                                  
    delta_r            = np.zeros_like(r)
    del_r              = r[1:] - r[:-1]
    delta_r[0]         = 2*del_r[0]
    delta_r[-1]        = 2*del_r[-1]
    delta_r[1:-1]      = (del_r[:-1]+ del_r[1:])/2
    delta_r            = to_jnumpy(delta_r) 

    bstei   = 1      # bottom surface trailing edge index 
    ustei   = -bstei # upper surface trailing edge index 

    if jnp.all(Omega == 0):
        res.p_pref_broadband                          = jnp.zeros((num_cpt,num_mic,num_rot,num_cf)) 
        res.SPL_prop_broadband_spectrum               = jnp.zeros_like(res.p_pref_broadband)
        res.SPL_prop_broadband_spectrum_dBA           = jnp.zeros_like(res.p_pref_broadband)
        res.SPL_prop_broadband_1_3_spectrum           = jnp.zeros((num_cpt,num_mic,num_rot,num_cf))
        res.SPL_prop_broadband_1_3_spectrum_dBA       = jnp.zeros((num_cpt,num_mic,num_rot,num_cf))
        res.p_pref_azimuthal_broadband                = jnp.zeros((num_cpt,num_mic,num_rot,num_azi,num_cf))
        res.p_pref_azimuthal_broadband_dBA            = jnp.zeros_like(res.p_pref_azimuthal_broadband)
        res.SPL_prop_azimuthal_broadband_spectrum     = jnp.zeros_like(res.p_pref_azimuthal_broadband)
        res.SPL_prop_azimuthal_broadband_spectrum_dBA = jnp.zeros_like(res.p_pref_azimuthal_broadband)
    else:
        delta        = jnp.zeros((num_cpt,num_mic,num_rot,num_sec,num_azi,num_cf,2)) #  control points ,  number rotors, number blades , number sections , sides of airfoil
        delta_star   = jnp.zeros_like(delta)
        dp_dx        = jnp.zeros_like(delta)
        tau_w        = jnp.zeros_like(delta)
        Ue           = jnp.zeros_like(delta)
        Theta        = jnp.zeros_like(delta)

        lower_surface_theta      = jnp.zeros((num_sec,num_azi))
        lower_surface_delta      = jnp.zeros_like(lower_surface_theta)
        lower_surface_delta_star = jnp.zeros_like(lower_surface_theta)
        lower_surface_cf         = jnp.zeros_like(lower_surface_theta)
        lower_surface_Ue         = jnp.zeros_like(lower_surface_theta)
        lower_surface_dp_dx      = jnp.zeros_like(lower_surface_theta)
        upper_surface_theta      = jnp.zeros_like(lower_surface_theta)
        upper_surface_delta      = jnp.zeros_like(lower_surface_theta)
        upper_surface_delta_star = jnp.zeros_like(lower_surface_theta)
        upper_surface_cf         = jnp.zeros_like(lower_surface_theta)
        upper_surface_Ue         = jnp.zeros_like(lower_surface_theta)
        upper_surface_dp_dx      = jnp.zeros_like(lower_surface_theta)

        if rotor.nonuniform_freestream: 

            # return the 1D Cl and CDval of shape (ctrl_pts, Nr)
            theta_ls         = jnp.zeros((num_cpt,num_sec,num_azi))
            delta_ls         = jnp.zeros((num_cpt,num_sec,num_azi))
            delta_star_ls    = jnp.zeros((num_cpt,num_sec,num_azi))
            Ue_Vinf_ls       = jnp.zeros((num_cpt,num_sec,num_azi))
            cf_ls            = jnp.zeros((num_cpt,num_sec,num_azi))
            dcp_dx_ls        = jnp.zeros((num_cpt,num_sec,num_azi))
            theta_us         = jnp.zeros((num_cpt,num_sec,num_azi))
            delta_us         = jnp.zeros((num_cpt,num_sec,num_azi))
            delta_star_us    = jnp.zeros((num_cpt,num_sec,num_azi))
            Ue_Vinf_us       = jnp.zeros((num_cpt,num_sec,num_azi))
            cf_us            = jnp.zeros((num_cpt,num_sec,num_azi))
            dcp_dx_us        = jnp.zeros((num_cpt,num_sec,num_azi))

            for i_azi in range(num_azi):
                for jj,airfoil in enumerate(airfoils):
                    bl                            = airfoil.polars.boundary_layer
                    local_aoa                     = alpha_blade[:,:,i_azi]
                    local_Re                      = Re_blade[:,:,i_azi]

                    theta_ls_data                 = interp2d(local_Re,local_aoa,bl.reynolds_numbers, bl.angle_of_attacks, bl.theta_lower_surface[:,:,bstei])     
                    delta_ls_data                 = interp2d(local_Re,local_aoa,bl.reynolds_numbers, bl.angle_of_attacks, bl.delta_lower_surface[:,:,bstei])        
                    delta_star_ls_data            = interp2d(local_Re,local_aoa,bl.reynolds_numbers, bl.angle_of_attacks, bl.delta_star_lower_surface[:,:,bstei])   
                    Ue_Vinf_ls_data               = interp2d(local_Re,local_aoa,bl.reynolds_numbers, bl.angle_of_attacks, bl.Ue_Vinf_lower_surface[:,:,bstei])      
                    cf_ls_data                    = interp2d(local_Re,local_aoa,bl.reynolds_numbers, bl.angle_of_attacks, bl.cf_lower_surface[:,:,bstei])           
                    dcp_dx_ls_data                = interp2d(local_Re,local_aoa,bl.reynolds_numbers, bl.angle_of_attacks, bl.dcp_dx_lower_surface[:,:,bstei])       
                    theta_us_data                 = interp2d(local_Re,local_aoa,bl.reynolds_numbers, bl.angle_of_attacks, bl.theta_upper_surface[:,:,ustei])        
                    delta_us_data                 = interp2d(local_Re,local_aoa,bl.reynolds_numbers, bl.angle_of_attacks, bl.delta_upper_surface[:,:,ustei])       
                    delta_star_us_data            = interp2d(local_Re,local_aoa,bl.reynolds_numbers, bl.angle_of_attacks, bl.delta_star_upper_surface[:,:,ustei])   
                    Ue_Vinf_us_data               = interp2d(local_Re,local_aoa,bl.reynolds_numbers, bl.angle_of_attacks, bl.Ue_Vinf_upper_surface[:,:,ustei])   
                    cf_us_data                    = interp2d(local_Re,local_aoa,bl.reynolds_numbers, bl.angle_of_attacks, bl.cf_upper_surface[:,:,ustei])          
                    dcp_dx_us_data                = interp2d(local_Re,local_aoa,bl.reynolds_numbers, bl.angle_of_attacks, bl.dcp_dx_upper_surface[:,:,ustei])      
                    locs                          = jnp.where(jnp.array(a_loc) == jj )

                    theta_ls      = theta_ls.at[:,locs,i_azi].set(theta_ls_data[:,locs])
                    delta_ls      = delta_ls.at[:,locs,i_azi].set(delta_ls_data[:,locs])
                    delta_star_ls = delta_star_ls.at[:,locs,i_azi].set(delta_star_ls_data[:,locs])
                    Ue_Vinf_ls    = Ue_Vinf_ls.at[:,locs,i_azi].set(Ue_Vinf_ls_data[:,locs])
                    cf_ls         = cf_ls.at[:,locs,i_azi].set(cf_ls_data[:,locs])
                    dcp_dx_ls     = dcp_dx_ls.at[:,locs,i_azi].set(dcp_dx_ls_data[:,locs])
                    theta_us      = theta_us.at[:,locs,i_azi].set(theta_us_data[:,locs])
                    delta_us      = delta_us.at[:,locs,i_azi].set(delta_us_data[:,locs])
                    delta_star_us = delta_star_us.at[:,locs,i_azi].set(delta_star_us_data[:,locs])
                    Ue_Vinf_us    = Ue_Vinf_us.at[:,locs,i_azi].set(Ue_Vinf_us_data[:,locs])
                    cf_us         = cf_us.at[:,locs,i_azi].set(cf_us_data[:,locs])
                    dcp_dx_us     = dcp_dx_us.at[:,locs,i_azi].set(dcp_dx_us_data[:,locs])

            blade_chords_3d           = jnp.tile(jnp.tile(blade_chords[None,:],(num_cpt,1))[:,:,None],(1,1,num_azi))
            dP_dX_ls                  = dcp_dx_ls*(0.5*rho_blade*U_blade**2)/blade_chords_3d
            dP_dX_us                  = dcp_dx_us*(0.5*rho_blade*U_blade**2)/blade_chords_3d

            lower_surface_theta       = theta_ls
            lower_surface_delta       = delta_ls
            lower_surface_delta_star  = delta_star_ls
            lower_surface_cf          = cf_ls
            lower_surface_Ue          = Ue_Vinf_ls*U_blade
            lower_surface_dp_dx       = dP_dX_ls
            upper_surface_theta       = theta_us
            upper_surface_delta       = delta_us
            upper_surface_delta_star  = delta_star_us
            upper_surface_cf          = cf_us
            upper_surface_Ue          = Ue_Vinf_us*U_blade
            upper_surface_dp_dx       = dP_dX_us

        else:
            theta_ls         = jnp.zeros((num_cpt,num_sec))
            delta_ls         = jnp.zeros((num_cpt,num_sec))
            delta_star_ls    = jnp.zeros((num_cpt,num_sec))
            Ue_Vinf_ls       = jnp.zeros((num_cpt,num_sec))
            cf_ls            = jnp.zeros((num_cpt,num_sec))
            dcp_dx_ls        = jnp.zeros((num_cpt,num_sec))
            theta_us         = jnp.zeros((num_cpt,num_sec))
            delta_us         = jnp.zeros((num_cpt,num_sec))
            delta_star_us    = jnp.zeros((num_cpt,num_sec))
            Ue_Vinf_us       = jnp.zeros((num_cpt,num_sec))
            cf_us            = jnp.zeros((num_cpt,num_sec))
            dcp_dx_us        = jnp.zeros((num_cpt,num_sec))
            local_aoa        = alpha_blade[:,:,0]
            local_Re         = Re_blade[:,:,0]

            for jj,airfoil in enumerate(airfoils):
                bl                       = airfoil.polars.boundary_layer
                theta_ls_data           = interp2d(local_Re,local_aoa,bl.reynolds_numbers, bl.angle_of_attacks, bl.theta_lower_surface[:,:,bstei])     
                delta_ls_data           = interp2d(local_Re,local_aoa,bl.reynolds_numbers, bl.angle_of_attacks, bl.delta_lower_surface[:,:,bstei])        
                delta_star_ls_data      = interp2d(local_Re,local_aoa,bl.reynolds_numbers, bl.angle_of_attacks, bl.delta_star_lower_surface[:,:,bstei])   
                Ue_Vinf_ls_data         = interp2d(local_Re,local_aoa,bl.reynolds_numbers, bl.angle_of_attacks, bl.Ue_Vinf_lower_surface[:,:,bstei])      
                cf_ls_data              = interp2d(local_Re,local_aoa,bl.reynolds_numbers, bl.angle_of_attacks, bl.cf_lower_surface[:,:,bstei])           
                dcp_dx_ls_data          = interp2d(local_Re,local_aoa,bl.reynolds_numbers, bl.angle_of_attacks, bl.dcp_dx_lower_surface[:,:,bstei])       
                theta_us_data           = interp2d(local_Re,local_aoa,bl.reynolds_numbers, bl.angle_of_attacks, bl.theta_upper_surface[:,:,ustei])        
                delta_us_data           = interp2d(local_Re,local_aoa,bl.reynolds_numbers, bl.angle_of_attacks, bl.delta_upper_surface[:,:,ustei])       
                delta_star_us_data      = interp2d(local_Re,local_aoa,bl.reynolds_numbers, bl.angle_of_attacks, bl.delta_star_upper_surface[:,:,ustei])   
                Ue_Vinf_us_data         = interp2d(local_Re,local_aoa,bl.reynolds_numbers, bl.angle_of_attacks, bl.Ue_Vinf_upper_surface[:,:,ustei])   
                cf_us_data              = interp2d(local_Re,local_aoa,bl.reynolds_numbers, bl.angle_of_attacks, bl.cf_upper_surface[:,:,ustei])          
                dcp_dx_us_data          = interp2d(local_Re,local_aoa,bl.reynolds_numbers, bl.angle_of_attacks, bl.dcp_dx_upper_surface[:,:,ustei])      

                locs                    = jnp.where(jnp.array(a_loc) == jj )

                theta_ls      = theta_ls.at[:,locs].set(theta_ls_data[:,locs])
                delta_ls      = delta_ls.at[:,locs].set(delta_ls_data[:,locs])
                delta_star_ls = delta_star_ls.at[:,locs].set(delta_star_ls_data[:,locs])
                Ue_Vinf_ls    = Ue_Vinf_ls.at[:,locs].set(Ue_Vinf_ls_data[:,locs])
                cf_ls         = cf_ls.at[:,locs].set(cf_ls_data[:,locs])
                dcp_dx_ls     = dcp_dx_ls.at[:,locs].set(dcp_dx_ls_data[:,locs])
                theta_us      = theta_us.at[:,locs].set(theta_us_data[:,locs])
                delta_us      = delta_us.at[:,locs].set(delta_us_data[:,locs])
                delta_star_us = delta_star_us.at[:,locs].set(delta_star_us_data[:,locs])
                Ue_Vinf_us    = Ue_Vinf_us.at[:,locs].set(Ue_Vinf_us_data[:,locs])
                cf_us         = cf_us.at[:,locs].set(cf_us_data[:,locs])
                dcp_dx_us     = dcp_dx_us.at[:,locs].set(dcp_dx_us_data[:,locs])

            blade_chords_2d           = jnp.tile(blade_chords[None,:],(num_cpt,1))
            dP_dX_ls                  = dcp_dx_ls*(0.5*rho_blade[:,:,0]*(U_blade[:,:,0]**2))/blade_chords_2d
            dP_dX_us                  = dcp_dx_us*(0.5*rho_blade[:,:,0]*(U_blade[:,:,0]**2))/blade_chords_2d

            lower_surface_theta       = jnp.tile(theta_ls[:,:,None],(1,1,num_azi))
            lower_surface_delta       = jnp.tile((blade_chords_2d*delta_ls)[:,:,None],(1,1,num_azi))
            lower_surface_delta_star  = jnp.tile(delta_star_ls[:,:,None],(1,1,num_azi))
            lower_surface_cf          = jnp.tile(cf_ls[:,:,None],(1,1,num_azi))
            lower_surface_Ue          = jnp.tile(Ue_Vinf_ls[:,:,None],(1,1,num_azi))*U_blade
            lower_surface_dp_dx       = jnp.tile(dP_dX_ls[:,:,None],(1,1,num_azi))
            upper_surface_theta       = jnp.tile(theta_us[:,:,None],(1,1,num_azi))
            upper_surface_delta       = jnp.tile((blade_chords_2d*delta_us)[:,:,None],(1,1,num_azi))
            upper_surface_delta_star  = jnp.tile(delta_star_us[:,:,None],(1,1,num_azi))
            upper_surface_cf          = jnp.tile(cf_us[:,:,None],(1,1,num_azi))
            upper_surface_Ue          = jnp.tile(Ue_Vinf_us[:,:,None],(1,1,num_azi))*U_blade
            upper_surface_dp_dx       = jnp.tile(dP_dX_us[:,:,None],(1,1,num_azi))
        # ------------------------------------------------------------
        # ****** TRAILING EDGE BOUNDARY LAYER PROPERTY CALCULATIONS  ******

        delta      = delta.at[:,:,:,:,:,:,0].set(jnp.tile(lower_surface_delta[:,None,None,:,:,None],(1,num_mic,num_rot,1,1,num_cf)))      # lower surface boundary layer thickness
        delta      = delta.at[:,:,:,:,:,:,1].set(jnp.tile(upper_surface_delta[:,None,None,:,:,None],(1,num_mic,num_rot,1,1,num_cf)))       # upper surface boundary layer thickness
        delta_star = delta_star.at[:,:,:,:,:,:,0].set(jnp.tile(lower_surface_delta_star[:,None,None,:,:,None],(1,num_mic,num_rot,1,1,num_cf)))  # lower surface displacement thickness
        delta_star = delta_star.at[:,:,:,:,:,:,1].set(jnp.tile(upper_surface_delta_star[:,None,None,:,:,None],(1,num_mic,num_rot,1,1,num_cf)))  # upper surface displacement thickness
        dp_dx      = dp_dx.at[:,:,:,:,:,:,0].set(jnp.tile(lower_surface_dp_dx[:,None,None,:,:,None],(1,num_mic,num_rot,1,1,num_cf)))       # lower surface pressure differential
        dp_dx      = dp_dx.at[:,:,:,:,:,:,1].set(jnp.tile(upper_surface_dp_dx[:,None,None,:,:,None],(1,num_mic,num_rot,1,1,num_cf)))       # upper surface pressure differential
        Ue         = Ue.at[:,:,:,:,:,:,0].set(jnp.tile(lower_surface_Ue[:,None,None,:,:,None],(1,num_mic,num_rot,1,1,num_cf)))          # lower surface boundary layer edge velocity
        Ue         = Ue.at[:,:,:,:,:,:,1].set(jnp.tile(upper_surface_Ue[:,None,None,:,:,None],(1,num_mic,num_rot,1,1,num_cf)))          # upper surface boundary layer edge velocity
        tau_w      = tau_w.at[:,:,:,:,:,:,0].set(jnp.tile((lower_surface_cf*(0.5*rho_blade*(U_blade**2)))[:,None,None,:,:,None],(1,num_mic,num_rot,1,1,num_cf)))        # lower surface wall shear stress
        tau_w      = tau_w.at[:,:,:,:,:,:,1].set(jnp.tile((upper_surface_cf*(0.5*rho_blade*(U_blade**2)))[:,None,None,:,:,None],(1,num_mic,num_rot,1,1,num_cf)))      # upper surface wall shear stress
        Theta      = Theta.at[:,:,:,:,:,:,0].set(jnp.tile(lower_surface_theta[:,None,None,:,:,None],(1,num_mic,num_rot,1,1,num_cf)))       # lower surface momentum thickness
        Theta      = Theta.at[:,:,:,:,:,:,1].set(jnp.tile(upper_surface_theta[:,None,None,:,:,None],(1,num_mic,num_rot,1,1,num_cf)))       # upper surface momentum thickness

        # Update dimensions for computation
        r         = jnp.tile(r[None,None,None,:,None,None],(num_cpt,num_mic,num_rot,1,num_azi,num_cf))
        c         = jnp.tile(blade_chords[None,None,None,:,None,None],(num_cpt,num_mic,num_rot,1,num_azi,num_cf))
        delta_r   = jnp.tile(delta_r[None,None,None,:,None,None],(num_cpt,num_mic,num_rot,1,num_azi,num_cf))
        M         = jnp.tile(M[:,None,None,None,None,None,:],(1,num_mic,num_rot,num_sec,num_azi,num_cf,1))  
        c_0       = jnp.tile(c_0[:,None,None,None,None,None,:],(1,num_mic,num_rot,num_sec,num_azi,num_cf,2))
        beta_sq   = jnp.tile(beta_sq[:,None,None,None,None,None,:],(1,num_mic,num_rot,num_sec,num_azi,num_cf,2))
        Omega     = jnp.tile(Omega[:,None,None,None,None,None,:],(1,num_mic,num_rot,num_sec,num_azi,num_cf,2))
        U_inf     = jnp.tile(U_inf[:,None,None,None,None,None,:],(1,num_mic,num_rot,num_sec,num_azi,num_cf,2))
        rho       = jnp.tile(rho[:,None,None,None,None,None,:],(1,num_mic,num_rot,num_sec,num_azi,num_cf,2))
        kine_visc = jnp.tile(kine_visc[:,None,None,None,None,None,:],(1,num_mic,num_rot,num_sec,num_azi,num_cf,2))

        X   = jnp.repeat(bspv.blade_section_coordinate_sys[:,:,:,:,:,:,0,:],2,axis = 6)
        Y   = jnp.repeat(bspv.blade_section_coordinate_sys[:,:,:,:,:,:,1,:],2,axis = 6)
        Z   = jnp.repeat(bspv.blade_section_coordinate_sys[:,:,:,:,:,:,2,:],2,axis = 6)

        # ------------------------------------------------------------
        # ****** BLADE MOTION CALCULATIONS ******
        # the rotational Mach number of the blade section
        omega   = jnp.tile((2*jnp.pi*frequency)[None,None,None,None,None,:,None],(num_cpt,num_mic,num_rot,num_sec,num_azi,1,2))
        r       = jnp.repeat(r[:,:,:,:,:,:,jnp.newaxis],2,axis = 6)
        c       = jnp.repeat(c[:,:,:,:,:,:,jnp.newaxis],2,axis = 6)/2
        delta_r = jnp.repeat(delta_r[:,:,:,:,:,:,jnp.newaxis],2,axis = 6)
        M       = jnp.repeat(M,2,axis = 6)
        R_s     = jnp.repeat(jnp.linalg.norm(bspv.blade_section_coordinate_sys,axis = 6),2,axis = 6)
        mu      = (omega/(1 +(Omega*r/c_0)*(X/R_s)))*M/(U_inf*beta_sq)

        # ------------------------------------------------------------
        # ****** LOADING TERM CALCULATIONS ******
        # equation 7
        epsilon       = X**2 + (beta_sq)*(Y**2 + Z**2)
        gamma         = jnp.sqrt(((mu/epsilon)**2)*(X**2 + beta_sq*(Z**2)))
        ss_1,cc_1     = jax2tf.call_tf(fresnel_tf)(2*((((omega/(1 +  (Omega*r/c_0)*(X/R_s))) /(0.8*U_inf)) /c) + (mu/c)*M + (gamma/c)))
        ss_2,cc_2     = jax2tf.call_tf(fresnel_tf)(2*((mu/c)*X/epsilon + (gamma/c)) )
        triangle      = (omega/(U_inf*c)) - (mu/c)*X/epsilon + (mu/c)*M
        norm_L_sq     = (1/triangle)*abs(jnp.exp(1j*2*triangle)*((1 - (1 + 1j)*(cc_1 - 1j*ss_1)) \
                        + ((jnp.exp(-1j*2*triangle))*(jnp.sqrt((((omega/(1 +  (Omega*r/c_0)*(X/R_s))) /(0.8*U_inf)) + mu*M + gamma)/(mu*X/epsilon +gamma))) \
                           *(1 + 1j)*(cc_2 - 1j*ss_2)) ))

        # ------------------------------------------------------------
        # ****** EMPIRICAL WALL PRESSURE SPECTRUM ******
        ones                     = jnp.ones_like(Theta)
        beta_c                   = (Theta/tau_w)*dp_dx 
        d                        = 4.76*((1.4/(delta/delta_star))**0.75)*(0.375*(3.7 + 1.5*beta_c) - 1)
        a                        = (2.82*((delta/delta_star)**2)*(jnp.power((6.13*((delta/delta_star)**(-0.75)) + d),(3.7 + 1.5*beta_c))))*\
                                   (4.2*((0.8*((beta_c + 0.5)**3/4))/(delta/delta_star)) + 1)
        d_star                   = d
        d_star[beta_c<0.5]       = jnp.maximum(ones,1.5*d)[beta_c<0.5]
        Phi_pp_expression        =  (jnp.maximum(a, (0.25*beta_c - 0.52)*a)*((omega*delta_star/Ue)**2))/(((4.76*((omega*delta_star/Ue)**0.75) \
                                    + d_star)**(3.7 + 1.5*beta_c))+ (jnp.power((8.8*(((delta/Ue)/(kine_visc/(((tau_w/rho)**0.5)**2)))**(-0.57))\
                                    *(omega*delta_star/Ue)),(jnp.minimum(3*ones,(0.139 + 3.1043*beta_c)) + 7)) ))
        Phi_pp                   = ((tau_w**2)*delta_star*Phi_pp_expression)/Ue

    
        inf_flag   = jnp.isinf(Phi_pp)
        Phi_pp     = Phi_pp.at[inf_flag].set(0) 
        nan_flag   = jnp.isnan(Phi_pp)
        Phi_pp     = Phi_pp.at[nan_flag].set(0) 
 
        # Power Spectral Density from each blade
        mult       = ((omega/c_0)**2)*(c**2)*delta_r*(1/(32*jnp.pi**2))*(B/(2*jnp.pi))
        int_x      = jnp.linspace(0,2*jnp.pi,num_azi)  
        S_pp       = mult[:,:,:,:,0,:,:]*jnp.trapz(((Z/(X**2 + (1-M**2)*(Y**2 + Z**2)))**2)*norm_L_sq*\
                                                  (1.6*(0.8*U_inf)/omega)*Phi_pp,x = int_x,axis = 4) 
            
        # Sound Pressure Level
        SPL                        = 10*jnp.log10((2*jnp.pi*abs(S_pp))/((p_ref)**2))  
        SPL_inf_flag               = jnp.isinf(SPL)
        SPL                        = SPL.at[SPL_inf_flag].set(0)  
        SPL_rotor                  = SPL_arithmetic(SPL_arithmetic(SPL, sum_axis = 5 ), sum_axis = 3 )
        
        # convert to 1/3 octave spectrum
        f = jnp.repeat(jnp.atleast_2d(frequency),num_cpt,axis = 0)
 
        res.SPL_prop_broadband_spectrum                   = SPL_rotor
        res.SPL_prop_broadband_spectrum_dBA               = A_weighting(SPL_rotor,frequency) 
        res.SPL_prop_broadband_1_3_spectrum               = SPL_harmonic_to_third_octave(SPL_rotor,f,settings)
        res.SPL_prop_broadband_1_3_spectrum_dBA           = SPL_harmonic_to_third_octave(A_weighting(SPL_rotor,frequency),f,settings) 
        res.SPL_prop_broadband_1_3_spectrum               = SPL_harmonic_to_third_octave(SPL_rotor,f,settings)
        
    return

def fresnel_tf(z):
    return fresnel_sin(z), fresnel_cos(z)
