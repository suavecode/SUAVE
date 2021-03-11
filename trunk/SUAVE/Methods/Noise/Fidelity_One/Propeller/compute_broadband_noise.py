# compute_broadband_noise.py
#
# Created:  Mar 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units , Data
import numpy as np
from scipy.special import jv 

from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.decibel_arithmetic import pressure_ratio_to_SPL_arithmetic
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.dbA_noise          import A_weighting
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry import import_airfoil_geometry
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_naca_4series import compute_naca_4series  

from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import pnl_noise
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import noise_tone_correction
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import epnl_noise
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import atmospheric_attenuation
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import noise_geometric
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import SPL_arithmetic
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import senel_noise
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import dbA_noise
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import SPL_harmonic_to_third_octave

## @ingroupMethods-Noise-Fidelity_One-Propeller
def compute_broadband_noise(ctrl_pt, p_idx, mic_loc,propeller,auc_opts,segment,S_r):

    # compute coordinated on all points on blade at control point , i
    X,Xp,Xe,Xr,Xpr,Xer,tau_r = compute_coordinates(segment,propeller,auc_opts,ctrl_pt,mic_loc,p_idx,S_r)       

    conditions             = segment.state.conditions
    microphone_location    = conditions.noise.microphone_locations
    angle_of_attack        = conditions.aerodynamics.angle_of_attack 
    velocity_vector        = conditions.frames.inertial.velocity_vector
    freestream             = conditions.freestream   

    a                      = freestream.speed_of_sound[ctrl_pt][0]       # speed of sound
    mu                     = freestream.dynamic_viscosity[ctrl_pt][0]       # speed of sound
    rho                    = freestream.density[ctrl_pt][0]              # air density 
    AoA                    = angle_of_attack[ctrl_pt][0]                 # vehicle angle of attack
    thrust_angle           = auc_opts.thrust_angle                 # propeller thrust angle
    
    omega                  = auc_opts.omega[ctrl_pt]                     # angular velocity      
    r_mn                   = auc_opts.disc_radial_distribution     # 2D radial location   
    psi_mn                 = auc_opts.disc_azimuthal_distribution  
    V_tot                  = auc_opts.mean_total_flow     
    blade_pitch            = auc_opts.blade_pitch 
    alpha_ask              = blade_pitch
    dL_dy                  = auc_opts.dL_dr   
    B                      = propeller.number_of_blades                             # number of propeller blades
    r                      = propeller.radius_distribution         # radial location    
    beta                   = propeller.twist_distribution          # twist distribution   
    c                      = propeller.chord_distribution                                  
    
    dim_radial             = np.shape(psi_mn)[1] 
    dim_azimuth            = np.shape(psi_mn)[2]   

 
    ## that the boundary layer displacement thicknesses on the two sides, as well as boundary layer thicknesses
    ## to be used below, are computed as functions of Rec and a* which were derived from detailed 
    ## near-wake flow measurements from the aforementioned isolated NACA 0012 blade section studies
    for B_idx in range(B): 
        for n_a in range(dim_azimuth):
            
            # 2 dimensiona radial distribution non dimensionalized
            alpha_tip     = blade_pitch[-1]
            alpha_p       = np.atleast_2d(blade_pitch)            
            alpha_p       = np.repeat(alpha_p,dim_azimuth, axis=1)   
            c_tip           = c[-1]
            dL_dy_ref       = 0 #CORRECT  Airfoil Tip Vortex Formulation Noise  
            alpha_prime_tip = (dL_dy[-1]/dL_dy_ref)*alpha_p[-1]            
            U_2d            = r_mn*omega
            Re_c            = c*U_2d*rho/mu   
            
            M              = U_2d/a   
            M_max          = (1 + 0.036*alpha_tip)*M   # maximum velocity in or about the vortex tear the trailing edge   
            phi            = 14*Units.degrees # solid angle between the bkade surfaces immediarely upstream of the trailing edge  # ref 27
            h              = 9E-3 # trailing edge thickness 
  
            L              = r[1:] - r[:-1] # blade segment width spanwise
            D_bar_h        =  
            r_e            = S     # observer distance form trailing edge
            
            
            # eqn 5 Airfoil self-noise and prediction
            delta0       = (10**(1.6569 - 0.9045*np.log(Re_c) + 0.0596*(np.log(Re_c))**2))*c 
            
            # eqn 6 Airfoil self-noise and prediction
            delta_star0  =  (10**(3.0187 - 1.5397*np.log(Re_c) + 0.1059*(np.log(Re_c))**2))*c  
            
            # eqn 7 Airfoil self-noise and prediction
            theta0       = (10**(0.2021 - 0.7079*np.log(Re_c) + 0.0404*(np.log(Re_c))**2))*c   
            
            # eqn 8 Airfoil self-noise and prediction
            delta_p       = (10**(-0.04175*alpha_ask + 0.0010*alpha_ask**2 ))*delta0 
            
            # eqn 9 Airfoil self-noise and prediction
            delta_star_p  =  (10**(-0.0432*alpha_ask  + 0.00113*alpha_ask**2 ))*delta_star0  
            
            # eqn 10 Airfoil self-noise and prediction
            theta_p       = (10**(-0.04508*alpha_ask + 0.000873*alpha_ask**2 ))*theta0          
            
            idx_1  = 
            idx_2  = 
            # eqn 11 Airfoil self-noise and prediction
            delta_s              = (10**(0.031114*alpha_ask) )*delta0 
            delta_s [idx_1]      =(0.0303*(10**(0.2336*alpha_ask) )*delta0)[idx_1] 
            delta_s [idx_2]      =(12*(10**(0.0258*alpha_ask))*delta0)[idx_2] 
            
            # eqn 12 Airfoil self-noise and prediction
            delta_star_s         =  (10**(0.0679*alpha_ask))*delta_star0  
            delta_star_s[idx_1]  =  (0.0162*(10**(0.03066*alpha_ask) )*delta_star0)[idx_1] 
            delta_star_s[idx_2]  =  (52.42*(10**(0.0258*alpha_ask))*delta_star0)[idx_2] 
            
            # eqn 13 Airfoil self-noise and prediction
            theta_s              = (10**(0.0559*alpha_ask) )*theta0          
            theta_s[idx_1]       = (0.0633*(10**(0.2157*alpha_ask) )*theta0)[idx_1]    
            theta_s[idx_2]       = (14.977(10**(0.0258*alpha_ask) )*theta0)[idx_2]    
            
            # average boundary layer displacement thickness of the pressure and suction sides any tailing edge  
            delta_star_avg = 0.5(delta_star_s + delta_star_p) 

            cos_zeta       = np.dot(Xpr,V_tot)/(np.linalg.norm(Xpr)*V_tot)
            
            # eqn 67 Airfoil self-noise and prediction
            if 0 < alpha_prime_tip and alpha_prime_tip < 2*Units.degrees: 
                l = (0.0230 + 0.0169*alpha_prime_tip)*c_tip
            else:
                l = (0.008*alpha_prime_tip)*c_tip   

            r_er                = np.linalg.norm(X_er) # or np.sqrt(z_er**2 +  y_er**2 + z_er**2)
            Theta_er            = np.arccos(x_er/r_er)
            Phi_er              = np.arccos(y_er/np.sqrt(y_er**2 + z_er**2)) 
                                
            D_bar_l             = ((np.sin(Theta_er/2)**2)*np.sin(Phi_er)**2)/(1 - M_tot*cos_zeta)**4
                                
            D_bar_h             =  (2*(np.sin(Theta_er/2)**2)*np.sin(Phi_er)**2)/(1 - M_tot*cos_zeta)**4 
  
            idx_3               = np.norm(alpha_ask) < 12.5*Units.degrees 
            idx_4               =  
            
            # Turbulent Boundary Layer - Trailing Edge noise Eqn 25  Airfoil self-noise and prediction 
            H_p                 = compute_H_p(f,delta_star_p,U,M,Re_c,Re_delta_star_p)  # ref 27  
            SPL_TBL_TEp         = ((delta_star_p*(M**5)*L*D_bar_h)/r_e**2)* H_p 
            SPL_TBL_TEp         = 0

            # suction side Eqn 26 Airfoil self-noise and prediction
            H_s                 = compute_H_s(f,delta_star_s,U,M,Re_c)# ref 27  
            SPL_TBL_TEs         = ((delta_star_p*(M**5)*L*D_bar_h)/r_e**2)* H_s  
            SPL_TBL_TEs         = 0

            # noise at angle of attack Eqn 27 Airfoil self-noise and prediction 
            H_alpha             =  compute_H_alpha(f,delta_star_s,alpha,M,Re_c)# ref 27  
            SBL_TEalpha         = ((delta_star_s*(M**5)*L*D_bar_h)/r_e**2)* H_alpha                  
            H_alpha             =  compute_H_alpha(f,delta_star_s,alpha,M,Re_c) 
            SBL_TEalpha[idx_3]  = ((delta_star_s*(M**5)*L*D_bar_l)/r_e**2)*H_alpha            
            SBL_TEalpha[idx_4]  = ((delta_star_s*(M**5)*L*D_bar_l)/r_e**2)*H_alpha 

            # summation of Turbulent Boundary Layer - Trailing Edge noise sources Eqn 25  Airfoil self-noise and prediction
            SPL_TBL_TE          = SPL_TBL_TEp + SPL_TBL_TEs + SBL_TEalpha  
            SPL_TBL_TE_cor      = SPL_TBL_TE/(beta**2) # glauret correction 
            
            # Laminar-boundary layer vortex shedding noise  
            # Eqn 53 Airfoil self-noise and prediction
            H_l                 =  
            SPL_LBL_VS          = ((delta_p*(M**5)*L*D_bar_h)/r_e**2)*H_l  

            # Blunt Trailing Edge Noise 
            # Eqn 70 Airfoil self-noise and prediction
            H_b                 =  
            SPL_BTE             = ((h*(M**5.5)*L*D_bar_h)/r_e**2)*H_b  

            # Tip noise 
            # Eqn 61 Airfoil self-noise and prediction
            H_t                 = 
            SPL_Tip             = (((M**2)*(M_max**3)*l*D_bar_h)/r_e**2)*H_t 

            # Addition of noise sources 
            SPL_self            = SPL_TBL_TE + SPL_LBL_VS + SPL_BTE + SPL_Tip

            # Blade Wake Interaction 
            M_tot               = np.linalg.norm(V_tot)/a
                                
            sigma**2            = (np.linalg.norm(X_r)**2)*(1 - M_tot*np.cos(zeta))**2
            zmu                 = (zeta*omega)/U
            or_as               = (omega*y_r)/(a*sigma**2)
            SPL_BWI             = (((omega*z_r)/(4*np.pi*a*sigma**2))**2)*SPL_LE_delta_p*((b*C)**2)*(2*L*zmu)/(zmu**2 + or_as**2)  

    SPL_BB     = np.sum(np.sum(4*(delta_phi/(2*np.pi))*(f_f0)(SPL_BWI + SPL_self)))

    SPL_BB_tot = 0

    return SPL_BB_tot

def compute_coordinates(segment,propeller,auc_opts,ctrl_pt,mic_loc,p_idx,S_r): 

    conditions            = segment.state.conditions
    mls                   = conditions.noise.microphone_locations 
    velocity_vector       = conditions.frames.inertial.velocity_vector
    freestream            = conditions.freestream     

    Vx                    = velocity_vector[ctrl_pt][0]  # x velocity of vehicle
    Vy                    = velocity_vector[ctrl_pt][1]  # y velocity of vehicle
    Vz                    = velocity_vector[ctrl_pt][2]  # z velocity of vehicle        
    psi_mn                = auc_opts.disc_azimuthal_distribution                      
    blade_pitch           = auc_opts.blade_pitch 
    B                     = propeller.number_of_blades                             # number of propeller blades
    r                     = propeller.radius_distribution                          # radial location     
    c                     = propeller.chord_distribution                           # blade chord     
    beta                  = propeller.twist_distribution                           # twist distribution  
    t                     = propeller.max_thickness_distribution                   # thickness distribution 
    MCA                   = propeller.mid_chord_aligment                           # Mid Chord Alighment
    prop_origin           = propeller.origin                   
    a_sec                 = propeller.airfoil_geometry          
    a_secl                = propeller.airfoil_polar_stations   
    dim_radial            = np.shape(psi_mn)[1] 
    dim_azimuth           = np.shape(psi_mn)[2]    
    a                     = freestream.speed_of_sound[ctrl_pt][0]                  # speed of sound 
    AoA                   = conditions.aerodynamics.angle_of_attack[ctrl_pt][0]    # vehicle angle of attack 
    thrust_angle          = auc_opts.thrust_angle                                  # propeller thrust angle  
    blade_angle           = np.linspace(0,2*np.pi,B+1)[:-1]  
    n_points              = 10

    # create empty arrays for storing geometry 
    X_mat    = np.zeros((B,dim_radial,dim_azimuth,3)) # radial , aximuth , airfoil pts  
    Xp_mat   = np.zeros_like(X_mat)  
    Xe_mat   = np.zeros_like(X_mat) 
    Xr_mat   = np.zeros_like(X_mat) 
    Xpr_mat  = np.zeros_like(X_mat) 
    Xer_mat  = np.zeros_like(X_mat) 

    for B_idx in range(B): 
        rot    = 1 # propeller.rotation[p_idx] 
        a_o    = 0
        flip_1 = (np.pi/2)   

        MCA_2d = np.repeat(np.atleast_2d(MCA).T,n_points,axis=1)
        b_2d   = np.repeat(np.atleast_2d(c).T  ,n_points,axis=1)
        t_2d   = np.repeat(np.atleast_2d(t).T  ,n_points,axis=1) 
        r_2d   = np.repeat(np.atleast_2d(r).T  ,n_points,axis=1)

        for n_a in range(dim_azimuth):  
            prop_plane_angle = blade_angle[B_idx] + psi_mn[ctrl_pt,0,n_a] 
            # get airfoil coordinate geometry   
            if a_sec != None:
                airfoil_data = import_airfoil_geometry(a_sec,npoints=n_points)   
                xpts         = np.take(airfoil_data.x_lower_surface,a_secl,axis=0)
                zpts         = np.take(airfoil_data.camber_coordinates,a_secl,axis=0) 
                max_t        = np.take(airfoil_data.thickness_to_chord,a_secl,axis=0) 

            else: 
                camber       = 0.02
                camber_loc   = 0.4
                thickness    = 0.10 
                airfoil_data = compute_naca_4series(camber, camber_loc, thickness,(n_points*2 - 2))                  
                xpts         = np.repeat(np.atleast_2d(airfoil_data.x_lower_surface) ,dim_radial,axis=0)
                zpts         = np.repeat(np.atleast_2d(airfoil_data.camber_coordinates) ,dim_radial,axis=0)
                max_t        = np.repeat(airfoil_data.thickness_to_chord,dim_radial,axis=0) 

            # store points of airfoil in similar format as Vortex Points (i.e. in vertices)   
            max_t2d = np.repeat(np.atleast_2d(max_t).T ,n_points,axis=1) 
            xp      = rot*(- MCA_2d + xpts*b_2d)  # x coord of airfoil
            yp      = r_2d*np.ones_like(xp)       # radial location        
            zp      = rot*zpts*(t_2d/max_t2d)     # former airfoil y coord 

            # -------------------------------------------------------------------
            # create matrices for X , Xp (x prime) and X_e 
            # -------------------------------------------------------------------
            # X occurs at the quarter chord so we will take the the corresponding point 
            X_mat_0      = np.zeros((dim_radial,4)) 
            X_mat_0[:,0] = xp[:,int(round(0.25*n_points))]
            X_mat_0[:,1] = yp[:,int(round(0.25*n_points))]
            X_mat_0[:,2] = zp[:,int(round(0.25*n_points))] 
            X_mat_0[:,3] = 1      

            # X_p occurs at the quarter chord so we will take the the corresponding point 
            Xp_mat_0      = np.zeros((dim_radial,4))  
            Xp_mat_0[:,0] = xp[:,int(round(0.25*n_points))]
            Xp_mat_0[:,1] = yp[:,int(round(0.25*n_points))]
            Xp_mat_0[:,2] = zp[:,int(round(0.25*n_points))]
            Xp_mat_0[:,3] = 1
            Xpr_mat_0     = Xp_mat_0

            # X_e occurs at the tip so we will take the the corresponding point and the end of the blade
            Xe_mat_0      = np.zeros((dim_radial,4)) 
            Xe_mat_0[:,0] = xp[:,-1]
            Xe_mat_0[:,1] = yp[:,-1]
            Xe_mat_0[:,2] = zp[:,-1]
            Xe_mat_0[:,3] = 1
            Xer_mat_0     = Xe_mat_0

            unit_vector_x0   = np.array([[-1*(-rot)] , [0], [0] , [0]]) # negative to match convention in paper
            unit_vector_y0   = np.array([[0] , [1], [0] , [0]])
            unit_vector_z0   = np.array([[0] , [0], [1] , [0]]) 

            # --------------------------------------------------------------------------------------
            # ROTATION & TRANSLATION MATRICES
            # --------------------------------------------------------------------------------------
            #  inverse rotation about y axis to create twist 
            rotation_1_1            = np.zeros((dim_radial,4,4))
            rotation_1_1[:,0,0]     = np.cos(rot*flip_1 - rot*beta)           
            rotation_1_1[:,0,2]     = - np.sin(rot*flip_1 - rot*beta)               
            rotation_1_1[:,1,1]     = 1
            rotation_1_1[:,2,0]     = np.sin(rot*flip_1 - rot*beta)    
            rotation_1_1[:,2,2]     = np.cos(rot*flip_1 - rot*beta) 
            rotation_1_1[:,3,3]     = 1 
            inv_rotation_1_1        = np.linalg.inv(rotation_1_1)   
            unit_vector_x1_1        = np.matmul(inv_rotation_1_1[0],unit_vector_x0)
            unit_vector_y1_1        = np.matmul(inv_rotation_1_1[0],unit_vector_y0)
            unit_vector_z1_1        = np.matmul(inv_rotation_1_1[0],unit_vector_z0)  

            #  inverse rotation about y axis to create twist AND pitch  
            rotation_1_2            = np.zeros((dim_radial,4,4))
            rotation_1_2[:,0,0]     = np.cos(rot*flip_1 - rot*blade_pitch)           
            rotation_1_2[:,0,2]     = - np.sin(rot*flip_1 - rot*blade_pitch)         
            rotation_1_2[:,1,1]     = 1
            rotation_1_2[:,2,0]     = np.sin(rot*flip_1 - rot*blade_pitch)    
            rotation_1_2[:,2,2]     = np.cos(rot*flip_1 - rot*blade_pitch) 
            rotation_1_2[:,3,3]     = 1 
            inv_rotation_1_2        = np.linalg.inv(rotation_1_2)   
            unit_vector_x1_2        = np.matmul(inv_rotation_1_2[0],unit_vector_x0)
            unit_vector_y1_2        = np.matmul(inv_rotation_1_2[0],unit_vector_y0)
            unit_vector_z1_2        = np.matmul(inv_rotation_1_2[0],unit_vector_z0)  


            # rotation about x axis to create azimuth locations 
            rotation_2            = np.zeros((dim_radial,4,4))
            rotation_2[:,0,0]     = 1                      
            rotation_2[:,1,1]     = np.cos(prop_plane_angle + rot*a_o)                                       
            rotation_2[:,1,2]     = -np.sin(prop_plane_angle + rot*a_o)                                
            rotation_2[:,2,1]     = np.sin(prop_plane_angle + rot*a_o)                                        
            rotation_2[:,2,2]     = np.cos(prop_plane_angle + rot*a_o)           
            rotation_2[:,3,3]     = 1 
            inv_rotation_2        = np.linalg.inv(rotation_2)  
            unit_vector_x2        = np.matmul(inv_rotation_2[0],unit_vector_x1_1)
            unit_vector_y2        = np.matmul(inv_rotation_2[0],unit_vector_y1_1)
            unit_vector_z2        = np.matmul(inv_rotation_2[0],unit_vector_z1_1) 
            unit_vector_x2_2      = np.matmul(inv_rotation_2[0],unit_vector_x1_2)
            unit_vector_y2_2      = np.matmul(inv_rotation_2[0],unit_vector_y1_2)
            unit_vector_z2_2      = np.matmul(inv_rotation_2[0],unit_vector_z1_2)                

            # rotation about y axis by thust angle 
            rotation_3            = np.zeros((dim_radial,4,4))
            rotation_3[:,0,0]     =  np.cos(thrust_angle)      
            rotation_3[:,0,2]     =  np.sin(thrust_angle)             
            rotation_3[:,1,1]     =  1
            rotation_3[:,2,0]     = -np.sin(thrust_angle) # -ve for inverse
            rotation_3[:,2,2]     =  np.cos(thrust_angle)
            rotation_3[:,3,3]     = 1 
            inv_rotation_3        = np.linalg.inv(rotation_3)  
            unit_vector_x3        = np.matmul(inv_rotation_3[0],unit_vector_x2)
            unit_vector_y3        = np.matmul(inv_rotation_3[0],unit_vector_y2)
            unit_vector_z3        = np.matmul(inv_rotation_3[0],unit_vector_z2) 
            unit_vector_x3_2      = np.matmul(inv_rotation_3[0],unit_vector_x2_2)
            unit_vector_y3_2      = np.matmul(inv_rotation_3[0],unit_vector_y2_2)
            unit_vector_z3_2      = np.matmul(inv_rotation_3[0],unit_vector_z2_2)                


            # translation to location on propeller on vehicle 
            translate_1            = np.repeat(np.atleast_3d(np.eye(4)).T,dim_radial,axis = 0) 
            translate_1[:,0,3]     = prop_origin[p_idx][0]   
            translate_1[:,1,3]     = prop_origin[p_idx][1]        
            translate_1[:,2,3]     = prop_origin[p_idx][2]     

            # rotation about y axis by angle of attack
            rotation_4            = np.zeros((dim_radial,4,4))
            rotation_4[:,0,0]     =  np.cos(AoA)      
            rotation_4[:,0,2]     = -np.sin(AoA)             
            rotation_4[:,1,1]     =  1      
            rotation_4[:,2,0]     =  np.sin(AoA)  
            rotation_4[:,2,2]     =  np.cos(AoA)
            rotation_4[:,3,3]     = 1                
            inv_rotation_4        = np.linalg.inv(rotation_4)
            velocity_unit_vector_x   = np.matmul(inv_rotation_4[0],unit_vector_x0)
            velocity_unit_vector_y   = np.matmul(inv_rotation_4[0],unit_vector_y0)
            velocity_unit_vector_z   = np.matmul(inv_rotation_4[0],unit_vector_z0)  
            Xcoord_sys_unit_vector_x = np.matmul(inv_rotation_4[0],unit_vector_x3)
            Xcoord_sys_unit_vector_y = np.matmul(inv_rotation_4[0],unit_vector_y3)
            Xcoord_sys_unit_vector_z = np.matmul(inv_rotation_4[0],unit_vector_z3) 
            Xcoord_sys_unit_vector_xe= np.matmul(inv_rotation_4[0],unit_vector_x3_2)
            Xcoord_sys_unit_vector_ye= np.matmul(inv_rotation_4[0],unit_vector_y3_2)
            Xcoord_sys_unit_vector_ze= np.matmul(inv_rotation_4[0],unit_vector_z3_2)                

            # translation to location on propeller on vehicle 
            translate_2            = np.repeat(np.atleast_3d(np.eye(4)).T,dim_radial,axis = 0) 
            translate_2[:,0,3]     = mls[ctrl_pt,mic_loc,0]  
            translate_2[:,1,3]     = mls[ctrl_pt,mic_loc,1]       
            translate_2[:,2,3]     = mls[ctrl_pt,mic_loc,2]         

            # create retarded coordinates  
            tau_r                    = S_r/a # propogation time  
            translate_2_r            = np.repeat(np.atleast_3d(np.eye(4)).T,dim_radial,axis = 0) 
            translate_2_r[:,0,3]     = mls[ctrl_pt,mic_loc,0] - Vx*tau_r  
            translate_2_r[:,1,3]     = mls[ctrl_pt,mic_loc,1] - Vy*tau_r       
            translate_2_r[:,2,3]     = mls[ctrl_pt,mic_loc,2] - Vz*tau_r  

            # ---------------------------------------------------------------------------------------------
            # ROTATE POINTS
            # ---------------------------------------------------------------------------------------------  
            # X' matrix 
            # need to validate ( may be X')
            Xp_mat_1       =  np.matmul(rotation_1_1,Xp_mat_0[...,None]).squeeze()      
            Xp_mat_2       =  np.matmul(rotation_2  ,Xp_mat_1[...,None]).squeeze()
            Xp_mat_3       =  np.matmul(rotation_3  ,Xp_mat_2[...,None]).squeeze()
            Xp_mat_4       =  np.matmul(translate_1 ,Xp_mat_3[...,None]).squeeze()
            Xp_mat_5       =  np.matmul(rotation_4  ,Xp_mat_4[...,None]).squeeze()
            Xp_mat_6       =  np.matmul(translate_2 ,Xp_mat_5[...,None]).squeeze() 
            Xp_mat_6       = - Xp_mat_6 # orient origin from  observe to from rotor  
            Xp_mat_7       = np.zeros_like(Xp_mat_6)
            Xp_mat_7[:,0]  = np.dot(Xp_mat_6[:,0:3],velocity_unit_vector_x[0:3,:])[:,0]/np.linalg.norm(velocity_unit_vector_x[0:3,:])
            Xp_mat_7[:,1]  = np.dot(Xp_mat_6[:,0:3],velocity_unit_vector_y[0:3,:])[:,0]/np.linalg.norm(velocity_unit_vector_y[0:3,:])
            Xp_mat_7[:,2]  = np.dot(Xp_mat_6[:,0:3],velocity_unit_vector_z[0:3,:])[:,0]/np.linalg.norm(velocity_unit_vector_z[0:3,:])       

            # X matrix 
            X_mat_1          = np.zeros_like(Xp_mat_6) # orient by twist, blade aximuth angle, thrust angle and aoa
            X_mat_1[:,0]     = np.dot(Xp_mat_6[:,0:3],Xcoord_sys_unit_vector_x[0:3,:])[:,0]/ np.linalg.norm(Xcoord_sys_unit_vector_x[0:3,:] )
            X_mat_1[:,1]     = np.dot(Xp_mat_6[:,0:3],Xcoord_sys_unit_vector_y[0:3,:])[:,0]/ np.linalg.norm(Xcoord_sys_unit_vector_y[0:3,:] )
            X_mat_1[:,2]     = np.dot(Xp_mat_6[:,0:3],Xcoord_sys_unit_vector_z[0:3,:])[:,0]/ np.linalg.norm(Xcoord_sys_unit_vector_z[0:3,:] )     

            # Xe matrix          
            Xe_mat_1       = np.matmul(rotation_1_2,Xe_mat_0[...,None]).squeeze()   
            Xe_mat_2       = np.matmul(rotation_2  ,Xe_mat_1[...,None]).squeeze()
            Xe_mat_3       = np.matmul(rotation_3  ,Xe_mat_2[...,None]).squeeze()
            Xe_mat_4       = np.matmul(translate_1 ,Xe_mat_3[...,None]).squeeze()
            Xe_mat_5       = np.matmul(rotation_4  ,Xe_mat_4[...,None]).squeeze()
            Xe_mat_6       = np.matmul(translate_2 ,Xe_mat_5[...,None]).squeeze()  
            Xe_mat_6       = - Xe_mat_6 # orient origin from  observe to from rotor  
            Xe_mat_7       = np.zeros_like(Xe_mat_6) # orient by twist, blade aximuth angle, thrust angle and aoa
            Xe_mat_7[:,0]  = np.dot(Xe_mat_6[:,0:3],Xcoord_sys_unit_vector_xe[0:3,:])[:,0]/np.linalg.norm(Xcoord_sys_unit_vector_xe[0:3,:])
            Xe_mat_7[:,1]  = np.dot(Xe_mat_6[:,0:3],Xcoord_sys_unit_vector_ye[0:3,:])[:,0]/np.linalg.norm(Xcoord_sys_unit_vector_ye[0:3,:])
            Xe_mat_7[:,2]  = np.dot(Xe_mat_6[:,0:3],Xcoord_sys_unit_vector_ze[0:3,:])[:,0]/np.linalg.norm(Xcoord_sys_unit_vector_ze[0:3,:])       

            # Retarded Matrices   
            # X'r matrix                  
            Xpr_mat_1       =  np.matmul(rotation_1_1  ,Xpr_mat_0[...,None]).squeeze()      
            Xpr_mat_2       =  np.matmul(rotation_2    ,Xpr_mat_1[...,None]).squeeze()
            Xpr_mat_3       =  np.matmul(rotation_3    ,Xpr_mat_2[...,None]).squeeze()
            Xpr_mat_4       =  np.matmul(translate_1   ,Xpr_mat_3[...,None]).squeeze()
            Xpr_mat_5       =  np.matmul(rotation_4    ,Xpr_mat_4[...,None]).squeeze()
            Xpr_mat_6       =  np.matmul(translate_2_r ,Xpr_mat_5[...,None]).squeeze() 
            Xpr_mat_6       = - Xpr_mat_6 # orient origin from  observe to from rotor                
            Xpr_mat_7       = np.zeros_like(Xpr_mat_6)
            Xpr_mat_7[:,0]  = np.dot(Xpr_mat_6[:,0:3],velocity_unit_vector_x[0:3,:])[:,0]/np.linalg.norm(velocity_unit_vector_x[0:3,:])
            Xpr_mat_7[:,1]  = np.dot(Xpr_mat_6[:,0:3],velocity_unit_vector_y[0:3,:])[:,0]/np.linalg.norm(velocity_unit_vector_y[0:3,:])
            Xpr_mat_7[:,2]  = np.dot(Xpr_mat_6[:,0:3],velocity_unit_vector_z[0:3,:])[:,0]/np.linalg.norm(velocity_unit_vector_z[0:3,:])       

            # Xr matrix 
            Xr_mat_1          = np.zeros_like(Xpr_mat_6) # orient by twist, blade aximuth angle, thrust angle and aoa
            Xr_mat_1[:,0]     = np.dot(Xpr_mat_6[:,0:3],Xcoord_sys_unit_vector_x[0:3,:])[:,0]/ np.linalg.norm(Xcoord_sys_unit_vector_x[0:3,:] )
            Xr_mat_1[:,1]     = np.dot(Xpr_mat_6[:,0:3],Xcoord_sys_unit_vector_y[0:3,:])[:,0]/ np.linalg.norm(Xcoord_sys_unit_vector_y[0:3,:] )
            Xr_mat_1[:,2]     = np.dot(Xpr_mat_6[:,0:3],Xcoord_sys_unit_vector_z[0:3,:])[:,0]/ np.linalg.norm(Xcoord_sys_unit_vector_z[0:3,:] )     

            # Xer matrix     
            Xer_mat_1       = np.matmul(rotation_1_2 ,Xer_mat_0[...,None]).squeeze()   
            Xer_mat_2       = np.matmul(rotation_2   ,Xer_mat_1[...,None]).squeeze()
            Xer_mat_3       = np.matmul(rotation_3   ,Xer_mat_2[...,None]).squeeze()
            Xer_mat_4       = np.matmul(translate_1  ,Xer_mat_3[...,None]).squeeze()
            Xer_mat_5       = np.matmul(rotation_4   ,Xer_mat_4[...,None]).squeeze()
            Xer_mat_6       = np.matmul(translate_2_r,Xer_mat_5[...,None]).squeeze()  
            Xer_mat_6       = - Xer_mat_6 # orient origin from  observe to from rotor  
            Xer_mat_7       = np.zeros_like(Xer_mat_6) # orient by twist, blade aximuth angle, thrust angle and aoa
            Xer_mat_7[:,0]  = np.dot(Xer_mat_6[:,0:3],Xcoord_sys_unit_vector_xe[0:3,:])[:,0]/np.linalg.norm(Xcoord_sys_unit_vector_xe[0:3,:])
            Xer_mat_7[:,1]  = np.dot(Xer_mat_6[:,0:3],Xcoord_sys_unit_vector_ye[0:3,:])[:,0]/np.linalg.norm(Xcoord_sys_unit_vector_ye[0:3,:])
            Xer_mat_7[:,2]  = np.dot(Xer_mat_6[:,0:3],Xcoord_sys_unit_vector_ze[0:3,:])[:,0]/np.linalg.norm(Xcoord_sys_unit_vector_ze[0:3,:]) 


            # ---------------------------------------------------------------------------------------------
            # STORE POINTS
            # ---------------------------------------------------------------------------------------------                
            X_mat[B_idx,:,n_a,0]  = X_mat_1[:,0]
            X_mat[B_idx,:,n_a,1]  = X_mat_1[:,1]
            X_mat[B_idx,:,n_a,2]  = X_mat_1[:,2]

            Xp_mat[B_idx,:,n_a,0] = Xp_mat_7[:,0]
            Xp_mat[B_idx,:,n_a,1] = Xp_mat_7[:,1]
            Xp_mat[B_idx,:,n_a,2] = Xp_mat_7[:,2]

            Xe_mat[B_idx,:,n_a,0] = Xe_mat_7[:,0]
            Xe_mat[B_idx,:,n_a,1] = Xe_mat_7[:,1]
            Xe_mat[B_idx,:,n_a,2] = Xe_mat_7[:,2] 

            Xr_mat[B_idx,:,n_a,0]  = Xr_mat_1[:,0]
            Xr_mat[B_idx,:,n_a,1]  = Xr_mat_1[:,1]
            Xr_mat[B_idx,:,n_a,2]  = Xr_mat_1[:,2]

            Xpr_mat[B_idx,:,n_a,0] = Xpr_mat_7[:,0]
            Xpr_mat[B_idx,:,n_a,1] = Xpr_mat_7[:,1]
            Xpr_mat[B_idx,:,n_a,2] = Xpr_mat_7[:,2]

            Xer_mat[B_idx,:,n_a,0] = Xer_mat_7[:,0]
            Xer_mat[B_idx,:,n_a,1] = Xer_mat_7[:,1]
            Xer_mat[B_idx,:,n_a,2] = Xer_mat_7[:,2]     

    return  X_mat, Xp_mat , Xe_mat, Xr_mat, Xpr_mat, Xer_mat , tau_r

#def compute_H_p(f,delta_star_p,U,M,Re_c,Re_delta_star_p):


    #return H_p


#def compute_H_s(f,delta_star_s,U,M,Re_c):


    #return H_s 

#def compute_H_alpha(f,delta_star_s,alpha,M,Re_c):



    #return H_alpha