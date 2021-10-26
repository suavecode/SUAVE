## @ingroupMethods-Noise-Fidelity_One-Propeller
# compute_broadband_noise.py
#
# Created:  Mar 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units , Data 
import numpy as np 
 
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.dbA_noise   import A_weighting  
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools             import SPL_harmonic_to_third_octave 
from SUAVE.Methods.Aerodynamics.Airfoil_Panel_Method.airfoil_analysis      import airfoil_analysis 
import matplotlib.pyplot as plt   
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_naca_4series \
     import  compute_naca_4series
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry\
     import import_airfoil_geometry
from scipy.special import fresnel

# ----------------------------------------------------------------------
# Frequency Domain Broadband Noise Computation
# ----------------------------------------------------------------------
## @ingroupMethods-Noise-Fidelity_One-Propeller
def compute_broadband_noise(freestream,angle_of_attack,position_vector,
                            velocity_vector,network,auc_opts,settings,res,source,BSR=100):
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
    ctrl_pts       = len(angle_of_attack)
    num_mic        = len(position_vector[0,:,0,1])
    num_prop       = len(position_vector[0,0,:,1]) 

    if source == 'lift_rotors': 
        propellers      = network.lift_rotors 
        propeller       = network.lift_rotors[list(propellers.keys())[0]]
    else:
        propellers      = network.propellers
        propeller       = network.propellers[list(propellers.keys())[0]] 
    
    # ----------------------------------------------------------------------------------
    # Trailing Edge Noise
    # ---------------------------------------------------------------------------------- 
     
    p_ref              = 2E-5               # referece atmospheric pressure
    c_0                = freestream.speed_of_sound  # speed of sound
    rho                = freestream.density  # air density 
    dyna_visc          = freestream.dynamic_viscosity
    kine_visc          = dyna_visc/rho # kinematic viscousity  
         
    AoA                = angle_of_attack         # vehicle angle of attack  
    thrust_angle       = propeller.orientation_euler_angles[1]     # propeller thrust angle
    alpha              =  0
    Re                 =   freestream.reynolds_number
    vehicle_position   = np.array([[0,0,1]])
    prop_origin        = np.array([[0,0,0]])      
    vehicle_position   = position_vector          # x component of position vector of propeller to microphone  
    vehicle_position   = vehicle_position[:,:,np.newaxis]   
    U_inf              = velocity_vector   
    U                  = U_inf
    B                  = propeller.number_of_blades          # number of propeller blades
    Omega              = auc_opts.omega               # angular velocity    
    R                  = propeller.radius_distribution       # radial location     
    c                  = propeller.chord_distribution        # blade chord    
    R_tip              = propeller.tip_radius                                                         
    t_c                = propeller.thickness_to_chord        # thickness to chord ratio
    MCA                = propeller.mid_chord_alignment       # Mid Chord Alighment   
    M                  = U_inf/c_0            
    x                  = position_vector[:,:,:,0]                               
    y                  = position_vector[:,:,:,1]
    z                  = position_vector[:,:,:,2]                                     
    omega              = auc_opts.omega[:,0]                                    # angular velocity        
    R                  = propeller.radius_distribution                          # radial location     
    num_sec            = len(R)
    c                  = propeller.chord_distribution                           # blade chord  
    beta               = propeller.twist_distribution                           # twist distribution   
    pi                 = np.pi 
    beta_sq            = 1 - M**2      
    N_r                = 1 
    Re                 = np.array([[1.5E6]])                                              #  CHECKED 
    alpha              = np.array([[6.]]) *Units.degrees                                            #  CHECKED 
    r_0                = np.linspace(-0.5,0.5,num_sec+1) # coordinated of blade corners 
    r                  = (r_0[:-1] + r_0[1:])/2 # centerpoints where noise/forces are computed 
    delta_r            = np.diff(r_0)                                                     #  CHECKED  
    c                  = np.ones_like(r)*0.4                                              #  CHECKED 
    chords             = np.ones(num_sec)*0.4                                             #  CHECKED   
    theta_0            = np.array([[0]])  # collective pitch angle that varies wih rotor thrust
    theta              = np.array([[6.]]) * Units.degrees   # twist angle
    phi                = np.array([[0]])   # azimuth angle  
    t                  = np.array([[0]]) # tite angle   
    beta_p             = np.array([[0]])   # blade flaping angle 
    t_v                = np.array([[0]]) # negative body angle    vehicle tilt angle between the vehicle hub plane and the geographical ground 
    t_r                = np.array([[0]]) # prop.orientation_euler_angles # rotor tilt angle between the rotor hub plane and the vehicle hub plane  

    # Update dimensions for computation   
    r            = vectorize_1(r,N_r,ctrl_pts,BSR) 
    c            = vectorize_1(c,N_r,ctrl_pts,BSR) 
    delta_r      = vectorize_1(delta_r,N_r,ctrl_pts,BSR) 
    theta_0      = vectorize_2(theta_0,N_r,num_sec,BSR) 
    theta        = vectorize_2(theta,N_r,num_sec,BSR) 
    M            = vectorize_2(M,N_r,num_sec,BSR)   
    phi          = vectorize_2(phi,N_r,num_sec,BSR)  
    beta_p       = vectorize_2(beta_p,N_r,num_sec,BSR)  
    t            = vectorize_2(t,N_r,num_sec,BSR)   
    t_v          = vectorize_2(t_v,N_r,num_sec,BSR)  
    t_r          = vectorize_2(t_r,N_r,num_sec,BSR)   
    c_0          = vectorize_3(c_0,N_r,num_sec,BSR)  
    beta_sq      = vectorize_3(beta_sq,N_r,num_sec,BSR) 
    Omega        = vectorize_3(Omega,N_r,num_sec,BSR)
    U_inf        = vectorize_3(U_inf,N_r,num_sec,BSR)
    kine_visc    = vectorize_3(kine_visc,N_r,num_sec,BSR) 
    POS_2        = vectorize_2(vehicle_position,N_r,num_sec,BSR) 
    M_hub        = vectorize_4(prop_origin,ctrl_pts,num_sec,BSR)    

    delta        = np.zeros((ctrl_pts,N_r,num_sec,BSR,2)) #  control points ,  number rotors, number blades , number sections , sides of airfoil   
    delta_star   = np.zeros_like(delta)
    dp_dx        = np.zeros_like(delta)
    tau_w        = np.zeros_like(delta)
    Ue           = np.zeros_like(delta)
    Theta        = np.zeros_like(delta)

    # ------------------------------------------------------------
    # ****** TRAILING EDGE BOUNDARY LAYER PROPERTY CALCULATIONS  ****** 
    for i in range(ctrl_pts) : 
        npanel               = 50
        Re_batch             = np.atleast_2d(np.ones(num_sec)*Re[i,0]).T
        AoA_batch            = np.atleast_2d(np.ones(num_sec)*alpha[i,0]).T       
        airfoil_geometry     = compute_naca_4series(0.0,0.0,0.12,npoints=npanel) 
        airfoil_stations     = [0] * num_sec
        AP                   = airfoil_analysis(airfoil_geometry,AoA_batch,Re_batch, npanel, batch_analysis = False, airfoil_stations = airfoil_stations)  

        # lower surface is 0, upper surface is 1 
        delta_star[i,:,:,:,0]   = vectorize_5(AP.delta_star[:,0],N_r,BSR)                         # lower surfacedisplacement thickness 
        delta_star[i,:,:,:,1]   = vectorize_5(AP.delta_star[:,-1],N_r,BSR)                        # upper surface displacement thickness  
        P                       = AP.Cp*(0.5*rho*U[i,0]**2)                                       
        blade_chords            = np.repeat(chords[:,np.newaxis],npanel, axis = 1)                   
        x_surf                  = AP.x*blade_chords                                                  
        dp_dx_surf              = np.diff(P)/np.diff(x_surf)                                 
        dp_dx[i,:,:,:,0]        = vectorize_5(dp_dx_surf[:,0],N_r,BSR)                            # lower surface pressure differential 
        dp_dx[i,:,:,:,1]        = vectorize_5(dp_dx_surf[:,-1],N_r,BSR)                           # upper surface pressure differential 
        U_e_lower_surf          = AP.Ue_Vinf[:,0]*U[i,0]
        U_e_upper_surf          = AP.Ue_Vinf[:,1]*U[i,0]
        Ue[i,:,:,:,0]           = vectorize_5(U_e_lower_surf,N_r,BSR)                             # lower surface boundary layer edge velocity 
        Ue[i,:,:,:,1]           = vectorize_5(U_e_upper_surf,N_r,BSR)                             # upper surface boundary layer edge velocity 
        tau_w[i,:,:,:,0]        = vectorize_5(AP.Cf[:,2]*(0.5*rho*U_e_lower_surf**2),N_r,BSR)     # lower surface wall shear stress
        tau_w[i,:,:,:,1]        = vectorize_5(AP.Cf[:,-2]*(0.5*rho*U_e_upper_surf**2),N_r,BSR)    # upper surface wall shear stress 
        Theta[i,:,:,:,0]        = vectorize_5(AP.theta[:,0],N_r,BSR)                              # lower surface momentum thickness     
        Theta[i,:,:,:,1]        = vectorize_5(AP.theta[:,-1],N_r,BSR)                             # upper surface momentum thickness  

    delta         = Theta*(3.15 + (1.72/((delta_star/Theta)- 1))) + delta_star 

    # ------------------------------------------------------------
    # ****** COORDINATE TRANSFOMRATIONS ****** 
    M_beta_p = np.zeros((ctrl_pts,N_r,num_sec,BSR,3,1))
    M_t      = np.zeros((ctrl_pts,N_r,num_sec,BSR,3,3))
    M_phi    = np.zeros((ctrl_pts,N_r,num_sec,BSR,3,3))
    M_theta  = np.zeros((ctrl_pts,N_r,num_sec,BSR,3,3))
    M_tv     = np.zeros((ctrl_pts,N_r,num_sec,BSR,3,3))

    M_tv[:,:,:,:,0,0]    = np.cos(t_v[:,:,:,:,0])
    M_tv[:,:,:,:,0,2]    = np.sin(t_v[:,:,:,:,0])
    M_tv[:,:,:,:,1,1]    = 1
    M_tv[:,:,:,:,2,0]    =-np.sin(t_v[:,:,:,:,0])
    M_tv[:,:,:,:,2,2]    = np.cos(t_v[:,:,:,:,0]) 

    # rotor hub position relative to center of aircraft 
    POS_1   = np.matmul(M_tv,(POS_2 + M_hub))  

    # twist angle matrix
    M_theta[:,:,:,:,0,0] =  np.cos(theta_0[:,:,:,:,0] + theta[:,:,:,:,0])            # CHECKED
    M_theta[:,:,:,:,0,2] =  np.sin(theta_0[:,:,:,:,0] + theta[:,:,:,:,0])
    M_theta[:,:,:,:,1,1] = 1
    M_theta[:,:,:,:,2,0] = -np.sin(theta_0[:,:,:,:,0] + theta[:,:,:,:,0])
    M_theta[:,:,:,:,2,2] =  np.cos(theta_0[:,:,:,:,0] + theta[:,:,:,:,0])

    # azimuth motion matrix
    M_phi[:,:,:,:,0,0] =  np.sin(phi[:,:,:,:,0])                                            # CHECKED      
    M_phi[:,:,:,:,0,1] = -np.cos(phi[:,:,:,:,0])  
    M_phi[:,:,:,:,1,0] =  np.cos(phi[:,:,:,:,0])  
    M_phi[:,:,:,:,1,1] =  np.sin(phi[:,:,:,:,0])     
    M_phi[:,:,:,:,2,2] = 1 

    # tilt motion matrix 
    M_t[:,:,:,:,0,0] =  np.cos(t[:,:,:,:,0])                                             # CHECKED
    M_t[:,:,:,:,0,2] =  np.sin(t[:,:,:,:,0]) 
    M_t[:,:,:,:,1,1] =  1 
    M_t[:,:,:,:,2,0] = -np.sin(t[:,:,:,:,0]) 
    M_t[:,:,:,:,2,2] =  np.cos(t[:,:,:,:,0]) 

    # flapping motion matrix
    M_beta_p[:,:,:,:,0,0]  = -r*np.sin(beta_p[:,:,:,:,0])*np.cos(phi[:,:,:,:,0]) # CHECKED
    M_beta_p[:,:,:,:,1,0]  = -r*np.sin(beta_p[:,:,:,:,0])*np.sin(phi[:,:,:,:,0])
    M_beta_p[:,:,:,:,2,0]  =  r*np.cos(beta_p[:,:,:,:,0])    

    # transformation of geographical global reference frame to the sectional local coordinate 
    mat0    = np.matmul(M_t,POS_1)  + M_beta_p                                        # CHECKED
    mat1    = np.matmul(M_phi,mat0)                                                   # CHECKED
    POS     = np.matmul(M_theta,mat1)                                                 # CHECKED 

    X   = np.repeat(POS[:,:,:,:,0,:],2,axis = 4)                                            # CHECKED
    Y   = np.repeat(POS[:,:,:,:,1,:],2,axis = 4)
    Z   = np.repeat(POS[:,:,:,:,2,:],2,axis = 4)
    X_1 = np.repeat(POS_1[:,:,:,:,0,:],2,axis = 4)
    Y_1 = np.repeat(POS_1[:,:,:,:,1,:],2,axis = 4)
    Z_1 = np.repeat(POS_1[:,:,:,:,2,:],2,axis = 4)    
    X_2 = np.repeat(POS_2[:,:,:,:,0,:],2,axis = 4)
    Y_2 = np.repeat(POS_2[:,:,:,:,1,:],2,axis = 4)
    Z_2 = np.repeat(POS_2[:,:,:,:,2,:],2,axis = 4)

    R_s = np.repeat(np.linalg.norm(POS,axis = 4),2,axis = 4) 

    # ------------------------------------------------------------
    # ****** BLADE MOTION CALCULATIONS ****** 
    # the rotational Mach number of the blade section 
    frequency = np.linspace(1E2,1E4, BSR)                                #  CHECKED AND VALIDATED
    omega     = 2*pi*frequency                                               #  CHECKED AND VALIDATED
    omega     = vectorize_6(omega,ctrl_pts,N_r,num_sec)                     #  CHECKED AND VALIDATED 
    r         = np.repeat(r[:,:,:,:,np.newaxis],2,axis = 4)                #  CHECKED AND VALIDATED 
    c         = np.repeat(c[:,:,:,:,np.newaxis],2,axis = 4)                #  CHECKED AND VALIDATED 
    delta_r   = np.repeat(delta_r[:,:,:,:,np.newaxis],2,axis = 4)          #  CHECKED AND VALIDATED 
    M         = np.repeat(M,2,axis = 4)                                    #  CHECKED AND VALIDATED 
    M_r       = Omega*r/c_0                                                #  CHECKED 
    epsilon   = X**2 + (beta_sq)*(Y**2 + Z**2)                             #  CHECKED
    U_c       = 0.8*U_inf                                                  #  CHECKED 
    k_x       = omega/U_inf                                                #  CHECKED
    l_r       = 1.6*U_c/omega                                              #  CHECKED   
    omega_d   = omega/(1 +  M_r*(X/R_s)) # dopler shifted frequency        #  CHECKED
    mu        = omega_d*M/(U_inf*beta_sq)  # omega_d*M/U_inf*beta_p        #  CHECKED
    bar_mu    = mu/(c/2)   # normalized by the semi chord                  #  CHECKED  
    bar_k_x   = k_x/(c/2)                                                  #  CHECKED

    # ------------------------------------------------------------
    # ****** LOADING TERM CALCULATIONS ******   
    # equation 7 
    triangle      = bar_k_x - bar_mu*X/epsilon + bar_mu*M                             #  CHECKED
    K             = omega_d/U_c                                                       #  CHECKED
    gamma         = np.sqrt(((mu/epsilon)**2)*(X**2 + beta_sq*(Z**2)))                #  CHECKED
    bar_K         = K /(c/2)                                                          #  CHECKED
    bar_gamma     = gamma/(c/2)                                                       #  CHECKED  
    ss_1, cc_1    = fresnel(2*(bar_K + bar_mu*M + bar_gamma))                            #  CHECKED                          
    E_star_1      = cc_1 - 1j*ss_1                                                        #  CHECKED 
    ss_2, cc_2    = fresnel(2*(bar_mu*X/epsilon + bar_gamma) )                           #  CHECKED                      
    E_star_2      = cc_2 - 1j*ss_2                                                        #  CHECKED 
    expression_A  = 1 - (1 + 1j)*E_star_1                                             #  CHECKED 
    expression_B  = (np.exp(-1j*2*triangle))*(np.sqrt((K + mu*M + gamma)/(mu*X/epsilon +gamma))) *(1 + 1j)*E_star_2    #  CHECKED 
    norm_L_sq     = (1/triangle)*abs(np.exp(1j*2*triangle)*(expression_A + expression_B ))                            #  CHECKED 


    # ------------------------------------------------------------
    # ****** EMPIRICAL WALL PRESSURE SPECTRUM ******  
    # equation 8 
    mu_tau              = (tau_w/rho)**0.5                                                          #  CHECKED AND VALIDATED
    ones                = np.ones_like(mu_tau)                                                      #  CHECKED AND VALIDATED
    R_T                 = (delta/Ue)/(kine_visc/(mu_tau**2))                                        #  CHECKED AND VALIDATED     
    beta_c              =  (Theta/tau_w)*dp_dx                                                      #  CHECKED AND VALIDATED                                       
    Delta               = delta/delta_star                                                          #  CHECKED AND VALIDATED
    e                   = 3.7 + 1.5*beta_c                                                          #  CHECKED AND VALIDATED
    d                   = 4.76*((1.4/Delta)**0.75)*(0.375*e - 1)                                    #  CHECKED AND VALIDATED                         
    PI                  = 0.8*((beta_c + 0.5)**3/4)                                                 #  CHECKED AND VALIDATED        
    a                   = (2.82*(Delta**2)*((6.13*(Delta**(-0.75)) + d)**e))*(4.2*(PI/Delta) + 1)   #  CHECKED AND VALIDATED
    h_star              = np.minimum(3*ones,(0.139 + 3.1043*beta_c)) + 7                            #  CHECKED AND VALIDATED
    d_star              = d                                                                         #  CHECKED AND VALIDATED 
    d_star[beta_c<0.5]  = np.maximum(ones,1.5*d)[beta_c<0.5]                                        #  CHECKED AND VALIDATED 
    expression_F        = (omega*delta_star/Ue)                                                     #  CHECKED AND VALIDATED
    expression_C        = np.maximum(a, (0.25*beta_c - 0.52)*a)*(expression_F**2)                   #  CHECKED AND VALIDATED
    expression_D        = (4.76*(expression_F**0.75) + d_star)**e                                   #  CHECKED AND VALIDATED 
    expression_E        = (8.8*(R_T**(-0.57))*expression_F)**h_star                                 #  CHECKED AND VALIDATED 
    Phi_pp_expression   =  expression_C/( expression_D + expression_E)                              #  CHECKED AND VALIDATED                           
    Phi_pp              = ((tau_w**2)*delta_star*Phi_pp_expression)/Ue                              #  CHECKED AND VALIDATED     

    # ------------------------------------------------------------
    # ****** DIRECTIVITY ******      
    #   equation A1 to A5 in Prediction of Urban Air Mobility Multirotor VTOL Broadband Noise Using UCD-QuietFly  
    l_x    = M_hub[:,:,:,:,0,:]
    l_y    = M_hub[:,:,:,:,1,:]
    l_z    = M_hub[:,:,:,:,2,:] 

    A4    = l_y + Y_2 - r*np.sin(beta_p)*np.sin(phi)
    A3    = (np.cos(t_r + t_v))*((np.cos(t_v))*(l_z + Z_2) - (np.sin(t_v))*(l_x + X_2))\
        - np.sin(t_r+ t_v)*((np.cos(t_v))*(l_x + X_2) + (np.sin(t_v))*l_z + Z_2) + r*np.cos(beta_p)
    A2    =  (np.cos(t_r + t_v))*((np.cos(t_v))*(l_x + X_2) + (np.sin(t_v))*(l_z + Z_2))\
        + np.sin(t_r+ t_v)*((np.cos(t_v))*(l_z + Z_2) - (np.sin(t_v))*l_x + X_2) - r*np.cos(phi)*np.cos(beta_p)
    A1    = (np.cos(theta_0+theta)*A3 + np.sin(theta_0+theta)*np.cos(beta_p)*A4 - np.sin(theta_0+theta)*np.sin(beta_p)*A2)**2
    D_phi = A1/( (np.sin(theta_0+theta)*A3 - np.cos(theta_0+theta)*np.cos(beta_p)*A4 \
                  + np.cos(theta_0+theta)*np.sin(beta_p)*A2**2)\
                 + (np.sin(beta_p)*A4 + np.cos(beta_p)*A2)**2)**2 

    # Acousic Power Spectrial Density from each blade - equation 6 
    S_pp   = ((omega/c_0 )**2)*c**2*delta_r*(1/(32*pi**2))*(B/(2*pi))*np.trapz(D_phi*norm_L_sq*l_r*Phi_pp) 

    # equation 9 
    SPL = 10*np.log10((2*pi*S_pp)/(p_ref**2)) 

    SPL_surf  = 10**(0.1*SPL[:,:,:,:,0]) + 10**(0.1*SPL[:,:,:,:,1]) # equation 10 inside brackets 
    SPL_blade = 10*np.log10(np.sum(SPL_surf,axis=2))  # equation 10 inside brackets 
    SPL_TE    = 10*np.log10(np.sum(SPL_blade,axis=1))  

 
    SPL_rotor_dBA    = A_weighting(SPL_TE,frequency) 
    
    res.p_pref_bb_dBA  = 10**(SPL_rotor_dBA /10)  
     
    # convert to 1/3 octave spectrum   
    res.SPL_prop_bb_spectrum = SPL_harmonic_to_third_octave(SPL_TE,f_v,settings)  
    
    return 

 

def vectorize_1(vec,N_r,ctrl_pts,BSR):
    # control points ,  number rotors, number blades , broadband section resolution, 1
    vec_x = np.repeat(np.repeat(np.repeat(np.atleast_2d(vec),N_r,axis = 0)[np.newaxis,:,:],ctrl_pts,axis = 0)[:,:,:,np.newaxis],BSR,axis =3)   
    return vec_x

def vectorize_2(vec,N_r,num_sec,BSR):
    # control points ,  number rotors, number blades , num sections , broadband section resolution,1
    vec_x = np.repeat(np.repeat(np.repeat(vec[:,np.newaxis,:],N_r,axis = 1)[:,:,np.newaxis],num_sec,axis = 2)[:,:,:,np.newaxis,:],BSR,axis = 3) 
    return  vec_x 

def vectorize_3(vec,N_r,num_sec,BSR):
    # control points ,  number rotors, number blades , num sections , broadband section resolution, num_surfaces(2)
    vec_x = np.repeat(np.repeat(np.repeat(np.repeat(vec[:,np.newaxis,:],N_r,axis = 1)[:,:,np.newaxis],num_sec,axis = 2),2,axis = 3)[:,:,:,np.newaxis,:],BSR,axis = 3)
    return vec_x

def vectorize_4(vec,ctrl_pts,num_sec,BSR): 
    # control points ,  number rotors, number blades , num sections , broadband section resolution, coordinates(3) , 1
    vec_x = np.repeat(np.repeat(np.repeat(vec[np.newaxis,:,:],ctrl_pts,axis = 0)[:,:,np.newaxis,:],num_sec,axis = 2)[:,:,:,np.newaxis,:],BSR,axis = 3)[:,:,:,:,:,np.newaxis]
    return vec_x 

def vectorize_5(vec,N_r,BSR):
    # number rotors, number blades , num sections , broadband section resolution
    vec_x = np.repeat(np.repeat(vec[np.newaxis,:],N_r,axis = 0)[:,:,np.newaxis],BSR,axis = 2)      
    return vec_x

def vectorize_6(vec,ctrl_pts,N_r,num_sec):
    # number rotors, number blades , num sections , broadband section resolution
    vec_x = np.repeat(np.repeat(np.repeat(np.repeat(vec[np.newaxis,:],num_sec,axis=0)[np.newaxis,:,:],N_r,axis=0)[np.newaxis,:,:,:],ctrl_pts,axis=0)[:,:,:,:,np.newaxis],2,axis=4)
    return vec_x


def vectorize_7(vec,ctrl_pts,N_r,num_sec):

    vec_x = np.repeat(np.repeat(np.repeat(np.repeat(vec[np.newaxis,:],num_sec,axis=0)[np.newaxis,:,:],N_r,axis=0)[np.newaxis,:,:,:],ctrl_pts,axis=0)[:,:,:,:,np.newaxis],2,axis=4)

    return vec_x