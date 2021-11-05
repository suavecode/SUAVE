## @ingroupMethods-Noise-Fidelity_One-Propeller
# compute_broadband_noise.py
#
# Created:  Mar 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
from SUAVE.Core import Units , Data 
import numpy as np 
 
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.dbA_noise                     import A_weighting  
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.SPL_harmonic_to_third_octave  import SPL_harmonic_to_third_octave
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.compute_source_coordinates    import vectorize_1,vectorize_2,vectorize_3,vectorize_4,vectorize_5,vectorize_8 
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_airfoil_boundary_layer_properties\
     import evaluate_boundary_layer_surrogates
from scipy.special import fresnel

# ----------------------------------------------------------------------
# Frequency Domain Broadband Noise Computation
# ----------------------------------------------------------------------
## @ingroupMethods-Noise-Fidelity_One-Propeller
def compute_broadband_noise(freestream,angle_of_attack,blade_section_position_vectors,
                            velocity_vector,network,auc_opts,settings,res,source):
    '''This computes the trailing edge noise compoment of broadband noise of a propeller or 
    rotor in the frequency domain
    
    Assumptions:
        UPDATE

    Source:
       UPDATE
    
    
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

    num_cpt         = len(angle_of_attack) 
    num_prop        = len(blade_section_position_vectors.blade_section_coordinate_sys[0,0,:,0,0,0,0,0]) 
    num_mic         = len(blade_section_position_vectors.blade_section_coordinate_sys[0,:,0,0,0,0,0,0])
    precision       = settings.floating_point_precision
    if source == 'lift_rotors':  
        propellers      = network.lift_rotors 
    else:
        propellers      = network.propellers
        
    propeller       = propellers[list(propellers.keys())[0]]   
    
    BSR       = settings.broadband_spectrum_resolution # broadband spectrum resolution    
    POS       = blade_section_position_vectors.blade_section_coordinate_sys   
    POS_2     = blade_section_position_vectors.vehicle_coordinate_sys           
    r         = blade_section_position_vectors.r                               
    beta_p    = blade_section_position_vectors.beta_p                          
    phi       = blade_section_position_vectors.phi                             
    alpha_eff = blade_section_position_vectors.alpha_eff                              
    t_v       = blade_section_position_vectors.t_v                             
    t_r       = blade_section_position_vectors.t_r                               
    M_hub     = blade_section_position_vectors.M_hub   
    
    # ----------------------------------------------------------------------------------
    # Trailing Edge Noise
    # ---------------------------------------------------------------------------------- 
    p_ref              = 2E-5                               # referece atmospheric pressure
    c_0                = freestream.speed_of_sound          # speed of sound
    rho                = freestream.density                 # air density 
    dyna_visc          = freestream.dynamic_viscosity
    kine_visc          = dyna_visc/rho                      # kinematic viscousity    
    alpha_blade        = auc_opts.disc_effective_angle_of_attack 
    Vt_2d              = auc_opts.disc_tangential_velocity  
    Va_2d              = auc_opts.disc_axial_velocity                
    blade_chords       = propeller.chord_distribution        # blade chord    
    r                  = propeller.radius_distribution        # radial location   
    num_sec            = len(r) 
    num_azi            = len(auc_opts.disc_effective_angle_of_attack[0,0,:])   
    U_blade            = np.sqrt(Vt_2d**2 + Va_2d **2)
    Re_blade           = U_blade*np.repeat(np.repeat(blade_chords[np.newaxis,:],num_cpt,axis=0)[:,:,np.newaxis],num_azi,axis=2)*\
                          np.repeat(np.repeat((rho/dyna_visc),num_sec,axis=1)[:,:,np.newaxis],num_azi,axis=2)
    rho_blade          = np.repeat(np.repeat(rho[:,np.newaxis],num_sec,axis=0)[:,:,np.newaxis],num_azi,axis=2)
    U_inf              = np.atleast_2d(np.linalg.norm(velocity_vector,axis=1)).T
    M                  = U_inf/c_0                                             
    B                  = propeller.number_of_blades          # number of propeller blades
    Omega              = auc_opts.omega                      # angular velocity   
    pi                 = np.pi 
    beta_sq            = 1 - M**2                                  
    delta_r            = np.zeros_like(r)
    del_r              = r[1:] - r[:-1]
    delta_r[0]         = 2*del_r[0]
    delta_r[-1]        = 2*del_r[-1]
    delta_r[1:-1]      =  (del_r[:-1]+ del_r[1:])/2 

    delta        = np.zeros((num_cpt,num_mic,num_prop,num_sec,num_azi,BSR,2)) #  control points ,  number rotors, number blades , number sections , sides of airfoil   
    delta_star   = np.zeros_like(delta)
    dp_dx        = np.zeros_like(delta)
    tau_w        = np.zeros_like(delta)
    Ue           = np.zeros_like(delta)
    Theta        = np.zeros_like(delta) 

    # ------------------------------------------------------------
    # ****** TRAILING EDGE BOUNDARY LAYER PROPERTY CALCULATIONS  ****** 
    bl_results = evaluate_boundary_layer_surrogates(propeller,alpha_blade,Re_blade)                         # lower surface is 0, upper surface is 1     
    delta_star[:,:,:,:,:,:,0]   = vectorize_2(bl_results.ls_delta_star,num_mic,num_prop,BSR)                # lower surfacedisplacement thickness 
    delta_star[:,:,:,:,:,:,1]   = vectorize_2(bl_results.us_delta_star,num_mic,num_prop,BSR)                # upper surface displacement thickness   
    dp_dx[:,:,:,:,:,:,0]        = vectorize_2(bl_results.ls_dcp_dx,num_mic,num_prop,BSR)                    # lower surface pressure differential 
    dp_dx[:,:,:,:,:,:,1]        = vectorize_2(bl_results.us_dcp_dx,num_mic,num_prop,BSR)                    # upper surface pressure differential 
    U_e_lower_surf              = bl_results.ls_Ue*U_blade[i,:,0]
    U_e_upper_surf              = bl_results.us_Ue*U_blade[i,:,0]
    Ue[:,:,:,:,:,:,0]           = vectorize_2(U_e_lower_surf,num_mic,num_prop,BSR)                          # lower surface boundary layer edge velocity 
    Ue[:,:,:,:,:,:,1]           = vectorize_2(U_e_upper_surf,num_mic,num_prop,BSR)                          # upper surface boundary layer edge velocity 
    tau_w[:,:,:,:,:,:,0]        = vectorize_2(bl_results.ls_cf*(0.5*rho_blade*U_e_lower_surf**2),num_mic,num_prop,BSR)    # lower surface wall shear stress
    tau_w[:,:,:,:,:,:,1]        = vectorize_2(bl_results.us_cf*(0.5*rho_blade* U_e_upper_surf**2),num_mic,num_prop,BSR)    # upper surface wall shear stress 
    Theta[:,:,:,:,:,:,0]        = vectorize_2(bl_results.ls_theta,num_mic,num_prop,BSR)                     # lower surface momentum thickness     
    Theta[:,:,:,:,:,:,1]        = vectorize_2(bl_results.us_theta,num_mic,num_prop,BSR)                     # upper surface momentum thickness  
    delta                       = Theta*(3.15 + (1.72/((delta_star/Theta)- 1))) + delta_star    
    
    # Update dimensions for computation      
    r         = vectorize_1(r,num_cpt,num_mic,num_prop,num_azi,BSR) 
    c         = vectorize_1(blade_chords,num_cpt,num_mic,num_prop,num_azi,BSR) 
    delta_r   = vectorize_1(delta_r,num_cpt,num_mic,num_prop,num_azi,BSR)   
    M         = vectorize_3(M,num_mic,num_prop,num_sec,num_azi,BSR)      
    c_0       = vectorize_4(c_0,num_mic,num_prop,num_sec,num_azi,BSR)  
    beta_sq   = vectorize_4(beta_sq,num_mic,num_prop,num_sec,num_azi,BSR) 
    Omega     = vectorize_4(Omega,num_mic,num_prop,num_sec,num_azi,BSR)
    U_inf     = vectorize_4(U_inf,num_mic,num_prop,num_sec,num_azi,BSR)
    rho       = vectorize_4(rho,num_mic,num_prop,num_sec,num_azi,BSR)
    kine_visc = vectorize_4(kine_visc,num_mic,num_prop,num_sec,num_azi,BSR)   

    X   = np.repeat(POS[:,:,:,:,:,:,0,:],2,axis = 6)                                    
    Y   = np.repeat(POS[:,:,:,:,:,:,1,:],2,axis = 6)
    Z   = np.repeat(POS[:,:,:,:,:,:,2,:],2,axis = 6)    
    X_2 = np.repeat(POS_2[:,:,:,:,:,:,0,:],2,axis = 6)
    Y_2 = np.repeat(POS_2[:,:,:,:,:,:,1,:],2,axis = 6)
    Z_2 = np.repeat(POS_2[:,:,:,:,:,:,2,:],2,axis = 6)

    R_s = np.repeat(np.linalg.norm(POS,axis = 6),2,axis = 6) 

    # ------------------------------------------------------------
    # ****** BLADE MOTION CALCULATIONS ****** 
    # the rotational Mach number of the blade section 
    frequency = np.linspace(1E2,1E4, BSR)                              
    omega   = 2*pi*frequency                                           
    omega   = vectorize_8(omega,num_mic,num_cpt,num_prop,num_azi,num_sec)     
    r       = np.repeat(r[:,:,:,:,:,:,np.newaxis],2,axis = 6)          
    c       = np.repeat(c[:,:,:,:,:,:,np.newaxis],2,axis = 6)          
    delta_r = np.repeat(delta_r[:,:,:,:,:,:,np.newaxis],2,axis = 6)    
    M       = np.repeat(M,2,axis = 6)                                  
    M_r     = Omega*r/c_0                                              
    epsilon = X**2 + (beta_sq)*(Y**2 + Z**2)                           
    U_c     = 0.8*U_inf                                                
    U_c     = 0.7*U_inf    # ONLY FOR VALIDATION
    k_x     = omega/U_inf                                              
    l_r     = 1.6*U_c/omega                                            
    l_r     = 1.*U_c/omega      # ONLY FOR VALIDATION
    omega_d = omega/(1 +  M_r*(X/R_s)) # dopler shifted frequency      
    mu      = omega_d*M/(U_inf*beta_sq)  # omega_d*M/U_inf*beta_p      
    bar_mu  = mu/(c/2)   # normalized by the semi chord                
    bar_k_x = k_x/(c/2)                                                

    # ------------------------------------------------------------
    # ****** LOADING TERM CALCULATIONS ******   
    # equation 7 
    triangle      = bar_k_x - bar_mu*X/epsilon + bar_mu*M                             
    K             = omega_d/U_c                                                       
    gamma         = np.sqrt(((mu/epsilon)**2)*(X**2 + beta_sq*(Z**2)))                
    bar_K         = K /(c/2)                                                          
    bar_gamma     = gamma/(c/2)                                                       
    ss_1, cc_1    = fresnel(2*(bar_K + bar_mu*M + bar_gamma))                                                
    E_star_1      = cc_1 - 1j*ss_1                                                        
    ss_2, cc_2    = fresnel(2*(bar_mu*X/epsilon + bar_gamma) )                                           
    E_star_2      = cc_2 - 1j*ss_2                                                        
    expression_A  = 1 - (1 + 1j)*E_star_1                                             
    expression_B  = (np.exp(-1j*2*triangle))*(np.sqrt((K + mu*M + gamma)/(mu*X/epsilon +gamma))) *(1 + 1j)*E_star_2     
    norm_L_sq     = (1/triangle)*abs(np.exp(1j*2*triangle)*(expression_A + expression_B ))                             
    
    # ------------------------------------------------------------
    # ****** EMPIRICAL WALL PRESSURE SPECTRUM ******  
    # equation 8 
    mu_tau              = (tau_w/rho)**0.5                                                          
    ones                = np.ones_like(mu_tau)                                                      
    R_T                 = (delta/Ue)/(kine_visc/(mu_tau**2))                                         
    beta_c              =  (Theta/tau_w)*dp_dx     
    Delta               = delta/delta_star                                                          
    e                   = 3.7 + 1.5*beta_c                                                          
    d                   = 4.76*((1.4/Delta)**0.75)*(0.375*e - 1)                                                         
    PI                  = 0.8*((beta_c + 0.5)**(3/4))                                                     
    a                   = (2.82*(Delta**2)*((6.13*(Delta**(-0.75)) + d)**e))*(4.2*(PI/Delta) + 1)   
    h_star              = np.minimum(3*ones,(0.139 + 3.1043*beta_c)) + 7                            
    d_star              = d                                                                         
    d_star[beta_c<0.5]  = np.maximum(ones,1.5*d)[beta_c<0.5]                                        
    expression_F        = (omega*delta_star/Ue)                                                     
    expression_C        = np.maximum(a, (0.25*beta_c - 0.52)*a)*(expression_F**2)                   
    expression_D        = (4.76*(expression_F**0.75) + d_star)**e                                   
    expression_E        = (8.8*(R_T**(-0.57))*expression_F)**h_star                                 
    Phi_pp_expression   =  expression_C/( expression_D + expression_E)                                                     
    Phi_pp              = ((tau_w**2)*delta_star*Phi_pp_expression)/Ue                                


    # ------------------------------------------------------------
    # ****** DIRECTIVITY ******      
    #   equation A1 to A5 in Prediction of Urban Air Mobility Multirotor VTOL Broadband Noise Using UCD-QuietFly 

    l_x    = M_hub[:,:,:,:,:,:,0,:]
    l_y    = M_hub[:,:,:,:,:,:,1,:]
    l_z    = M_hub[:,:,:,:,:,:,2,:] 

    A4    = l_y + Y_2 - r*np.sin(beta_p)*np.sin(phi)
    A3    = (np.cos(t_r + t_v))*((np.cos(t_v))*(l_z + Z_2) - (np.sin(t_v))*(l_x + X_2))\
        - np.sin(t_r+ t_v)*((np.cos(t_v))*(l_x + X_2) + (np.sin(t_v))*l_z + Z_2) + r*np.cos(beta_p)
    A2    =  (np.cos(t_r + t_v))*((np.cos(t_v))*(l_x + X_2) + (np.sin(t_v))*(l_z + Z_2))\
        + np.sin(t_r+ t_v)*((np.cos(t_v))*(l_z + Z_2) - (np.sin(t_v))*l_x + X_2) - r*np.cos(phi)*np.cos(beta_p)
    A1    = (np.cos( alpha_eff)*A3 + np.sin( alpha_eff)*np.cos(beta_p)*A4 - np.sin( alpha_eff)*np.sin(beta_p)*A2)**2
    D_phi = A1/( (np.sin( alpha_eff)*A3 - np.cos( alpha_eff)*np.cos(beta_p)*A4 \
                  + np.cos( alpha_eff)*np.sin(beta_p)*A2**2)\
                 + (np.sin(beta_p)*A4 + np.cos(beta_p)*A2)**2)**2 

    # Acousic Power Spectrial Density from each blade - equation 6 
    mult = ((omega/c_0 )**2)*c**2*delta_r*(1/(32*pi**2))*(B/(2*pi))
    S_pp   = mult[:,:,:,:,0,:,:]*np.trapz(D_phi*norm_L_sq*l_r*Phi_pp,axis = 4)

    # equation 9 
    SPL = 10*np.log10((2*pi*S_pp)/((p_ref)**2))
    #SPL[SPL<0] = 0 # CHECK!!!!!!!

    SPL_surf  = 10**(0.1*SPL[:,:,:,:,:,0]) + 10**(0.1*SPL[:,:,:,:,:,1]) # equation 10 inside brackets 
    SPL_blade = 10*np.log10(np.sum(SPL_surf,axis=3))  # equation 10 inside brackets 
    
    SPL_TE    = 10*np.log10(np.sum(SPL_blade,axis=2))   
 
    SPL_rotor_dBA    = A_weighting(SPL_TE,frequency) 
    
    res.p_pref_bb_dBA  = 10**(SPL_rotor_dBA /10)  
     
    # convert to 1/3 octave spectrum   
    res.SPL_prop_bb_spectrum = SPL_harmonic_to_third_octave(SPL_TE,frequency,settings)  
    
    return 

 