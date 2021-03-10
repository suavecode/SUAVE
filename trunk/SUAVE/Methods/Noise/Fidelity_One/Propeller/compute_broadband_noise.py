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
def compute_broadband_noise(i, p_idx, mic_loc,propeller,auc_opts,segment,S_r):
     
     X , Xp , Xe, Xr,Xpr,  Xer , tau_r = compute_coordinates(segment,propeller,auc_opts,i,mic_loc,p_idx,S_r)      
     
     conditions             = segment.state.conditions
     microphone_location    = conditions.noise.microphone_locations
     angle_of_attack        = conditions.aerodynamics.angle_of_attack 
     velocity_vector        = conditions.frames.inertial.velocity_vector
     freestream             = conditions.freestream   
     
     a                      = freestream.speed_of_sound[ctrl_pt][0]       # speed of sound
     rho                    = freestream.density[ctrl_pt][0]              # air density 
     AoA                    = angle_of_attack[ctrl_pt][0]                 # vehicle angle of attack 
     thrust_angle           = auc_opts.thrust_angle                 # propeller thrust angle
     alpha                  = AoA + thrust_angle     
     omega                  = auc_opts.omega[ctrl_pt]                     # angular velocity      
     r_mn                   = auc_opts.disc_radial_distribution     # 2D radial location   
     psi_mn                 = auc_opts.disc_azimuthal_distribution  
     V_tot                  = auc_opts.mean_total_flow                  
     r                      = propeller.radius_distribution         # radial location    
     beta                   = propeller.twist_distribution          # twist distribution   
                                                                   
     dim_radial             = np.shape(psi_mn)[0]
     dim_azimuth            = np.shape(psi_mn)[1]  
     
     # 2 dimensiona radial distribution non dimensionalized
     alpha_p       = np.atleast_2d(beta)            
     alpha_p       = np.repeat(alpha_p,dim_azimuth, axis=1)  
     
        
     #cos_zeta       = np.dot(X_prime_r,V_tot)/(np.linalg.norm(X_prime_r)*V_tot)
     #l              =  # spanwize extent of the vortex formation at the trailing edge 
     #alpha_eff_tip  = 
     #M_max          = alpha_eff_tip , M   # maximum velocity in or about the vortex tear the trailing edge alpha_eff_tip and M #
     #phi            = 14*Units.degrees # solid angle between the bkade surfaces immediarely upstream of the trailing edge  # ref 27
     #h              = 9E-3 # trailing edge thickness 
                    
     #delta_star_avg = # average boundary layer displacement thickness of the pressure and suction sides any tailing edge 
     #U              = # flow normal to the span (2 D array from hub to tip and azimuth) 
     #delta_star_p   = # displacement thickness pressure side   # ref 27 
     #delta_star_s   = # dispacement thickness suction side  # ref 27  
     #L              = r[1:] - r[:-1] # blade segment width spanwise
     #M              = U/a   
     #D_bar_h        =  
     #r_e            = S     # observer distance form trailing edge
     
     #r_er     = np.linalg.norm(X_er) # or np.sqrt(z_er**2 +  y_er**2 + z_er**2)
     #Theta_er = np.arccos(x_er/r_er)
     #Phi_er   = np.arccos(y_er/np.sqrt(y_er**2 + z_er**2)) 
     
     #D_bar_l = ((np.sin(Theta_er/2)**2)*np.sin(Phi_er)**2)/(1 - M_tot*cos_zeta)**4
     
     #D_bar_h =  (2*(np.sin(Theta_er/2)**2)*np.sin(Phi_er)**2)/(1 - M_tot*cos_zeta)**4
     
     #U = omega*r_2d
     ## that the boundary layer displacement thicknesses on the two sides, as well as boundary layer thicknesses
     ## to be used below, are computed as functions27 of Rec and a* which were derived from detailed 
     ## near-wake flow measurements from the aforementioned isolated NACA 0012 blade section studies
     #for b in range(B)
         #for m in range(dim_radial):
              #for n in range(dim_azimuth):
                  #alpha_star = 
                  #if np.norm(alpha_star) < 12.5*Units.degrees or   : 
                       ## Turbulent Boundary Layer - Trailing Edge noise
                       ###  pressure side 
                       #G_TBL_TEp      = 0
                       
                       ### suction side
                       #G_TBL_TEs      = 0
                       
                       ### noise at angle of attack 
                       #H_alpha        = compute_H_alpha(f,delta_star_s,alpha,M,Re_c) 
                       #T_TBL_TEalpha  = ((delta_star_s*(M**5)*L*D_bar_l)/r_e**2)* H_alpha            
                  #else:
                       ## Turbulent Boundary Layer - Trailing Edge noise
                       ###  pressure side
                       #H_p            = compute_H_p(f,delta_star_p,U,M,Re_c,Re_delta_star_p)  # ref 27  
                       #G_TBL_TEp      = ((delta_star_p*(M**5)*L*D_bar_h)/r_e**2)* H_p 
                       
                       ### suction side
                       #H_s            = compute_H_s(f,delta_star_s,U,M,Re_c)# ref 27  
                       #G_TBL_TEs      = ((delta_star_p*(M**5)*L*D_bar_h)/r_e**2)* H_s  
                       
                       ### noise at angle of attack 
                       #H_alpha        = compute_H_alpha(f,delta_star_s,alpha,M,Re_c)  # ref 27  
                       #T_TBL_TEalpha  = ((delta_star_s*(M**5)*L*D_bar_h)/r_e**2)* H_alpha  
                       
                  ## summation of Turbulent Boundary Layer - Trailing Edge noise sources 
                  #G_TBL_TE = G_TBL_TEp + G_TBL_TEs + T_TBL_TEalpha  
                  #G_TBL_TE_cor =  G_TBL_TE/(beta**2) # glauret correction 
                  ## Laminar-boundary layer vortex shedding noise 
                  #H_l      =  
                  #G_LBL_VS = ((delta_p*(M**5)*L*D_bar_h)/r_e**2)* H_l  
                  
                  ## Blunt Trailing Edge Noise 
                  #H_b     =  
                  #G_BTE   = ((h*(M**5.5)*L*D_bar_h)/r_e**2)* H_b  
                  
                  ## Tip noise 
                  #H_t    = 
                  #G_Tip  = (((M**2)*(M_max**3)*l*D_bar_h)/r_e**2)* H_t 
                  
                  ## Addition of noise sources 
                  #G_self = G_TBL_TE + G_LBL_VS + G_BTE + G_Tip
                  
                  ## Blade Wake Interaction 
                  #M_tot    = np.linalg.norm(V_tot)/a
                  
                  #sigma**2 = (np.linalg.norm(X_r)**2)*(1 - M_tot*np.cos(zeta))**2
                  #zmu      = (zeta*omega)/U
                  #or_as    = (omega*y_r)/(a*sigma**2)
                  #G_BWI    = (((omega*z_r)/(4*np.pi*a*sigma**2))**2)*G_LE_delta_p*((b*C)**2)*(2*L*zmu)/(zmu**2 + or_as**2)  
              
         #G_BB     = np.sum(np.sum(4*(delta_phi/(2*np.pi))*(f_f0)(G_BWI + G_self)))
         
     G_BB_tot = 0
     
     return G_BB_tot

def compute_coordinates(segment,propeller,auc_opts,ctrl_pt,mic_loc,p_idx,S_r): 
     
     conditions            = segment.state.conditions
     mls                   = conditions.noise.microphone_locations 
     velocity_vector       = conditions.frames.inertial.velocity_vector
     freestream            = conditions.freestream     

     Vx                    = velocity_vector[ctrl_pt][0]  # x velocity of vehicle
     Vy                    = velocity_vector[ctrl_pt][1]  # y velocity of vehicle
     Vz                    = velocity_vector[ctrl_pt][2]  # z velocity of vehicle        
     psi_mn                = auc_opts.disc_azimuthal_distribution         
     V_tot                 = auc_opts.mean_total_flow                                
     blade_pitch           = auc_opts.blade_pitch 
     B                     = propeller.number_of_blades                             # number of propeller blades
     r                     = propeller.radius_distribution                          # radial location     
     c                     = propeller.chord_distribution                           # blade chord    
     R_tip                 = propeller.tip_radius                                   
     beta                  = propeller.twist_distribution                           # twist distribution  
     t                     = propeller.max_thickness_distribution                   # thickness distribution 
     MCA                   = propeller.mid_chord_aligment                           # Mid Chord Alighment
     prop_origin           = propeller.origin                   
     a_sec                 = propeller.airfoil_geometry          
     a_secl                = propeller.airfoil_polar_stations   
     dim_radial            = np.shape(psi_mn)[1] 
     dim_azimuth           = np.shape(psi_mn)[2]    
     a                     = freestream.speed_of_sound[ctrl_pt][0]                  # speed of sound 
     AoA                   = -conditions.aerodynamics.angle_of_attack[ctrl_pt][0]   # vehicle angle of attack 
     thrust_angle          = -auc_opts.thrust_angle                                 # propeller thrust angle 
 
     blade_angle   = np.linspace(0,2*np.pi,B+1)[:-1]  
     n_points      = 20
 
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
               X_mat_0[:,0] = -xp[:,int(round(0.25*n_points))]
               X_mat_0[:,1] = -yp[:,int(round(0.25*n_points))]
               X_mat_0[:,2] = -zp[:,int(round(0.25*n_points))] 
               X_mat_0[:,3] = 1
               Xr_mat_0     = X_mat_0        
               
               # X_p occurs at the quarter chord so we will take the the corresponding point 
               Xp_mat_0      = np.zeros((dim_radial,4))  
               Xp_mat_0[:,0] = -xp[:,int(round(0.25*n_points))]
               Xp_mat_0[:,1] = -yp[:,int(round(0.25*n_points))]
               Xp_mat_0[:,2] = -zp[:,int(round(0.25*n_points))]
               Xp_mat_0[:,3] = 1
               Xpr_mat_0     = Xp_mat_0
               
               # X_e occurs at the tip so we will take the the corresponding point and the end of the blade
               Xe_mat_0      = np.zeros((dim_radial,4)) 
               Xe_mat_0[:,0] = -xp[:,-1]
               Xe_mat_0[:,1] = -yp[:,-1]
               Xe_mat_0[:,2] = -zp[:,-1]
               Xe_mat_0[:,3] = 1
               Xer_mat_0     = Xe_mat_0
               
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
               rotation_1_1[:,3,3]     =1
               
               inv_rotation_1_1        = np.zeros((dim_radial,4,4))
               inv_rotation_1_1[:,0,0] = np.cos(rot*flip_1 - rot*beta)            
               inv_rotation_1_1[:,0,2] = np.sin(rot*flip_1 - rot*beta)            
               inv_rotation_1_1[:,1,1] = 1
               inv_rotation_1_1[:,2,0] = -np.sin(rot*flip_1 - rot*beta)    # -ve for inverse   
               inv_rotation_1_1[:,2,2] = np.cos(rot*flip_1 - rot*beta) 
               inv_rotation_1_1[:,3,3] = 1
               
               #  inverse rotation about y axis to create twist AND pitch  
               rotation_1_2            = np.zeros((dim_radial,4,4))
               rotation_1_2[:,0,0]     = np.cos(rot*flip_1 - rot*blade_pitch)           
               rotation_1_2[:,0,2]     = - np.sin(rot*flip_1 - rot*blade_pitch)         
               rotation_1_2[:,1,1]     = 1
               rotation_1_2[:,2,0]     = np.sin(rot*flip_1 - rot*blade_pitch)    
               rotation_1_2[:,2,2]     = np.cos(rot*flip_1 - rot*blade_pitch) 
               rotation_1_2[:,3,3]     =1
               
               inv_rotation_1_2        = np.zeros((dim_radial,4,4))
               inv_rotation_1_2[:,0,0] = np.cos(rot*flip_1 - rot*blade_pitch)  
               inv_rotation_1_2[:,0,2] = np.sin(rot*flip_1 - rot*blade_pitch)                
               inv_rotation_1_2[:,1,1] = 1
               inv_rotation_1_2[:,2,0] = - np.sin(rot*flip_1 - rot*blade_pitch)   # -ve for inverse   
               inv_rotation_1_2[:,2,2] = np.cos(rot*flip_1 - rot*blade_pitch)  
               inv_rotation_1_2[:,3,3] =1
               

               # rotation about x axis to create azimuth locations 
               rotation_2            = np.zeros((dim_radial,4,4))
               rotation_2[:,0,0]     = 1                      
               rotation_2[:,1,1]     = np.cos(prop_plane_angle + rot*a_o)                                       
               rotation_2[:,1,2]     = -np.sin(prop_plane_angle + rot*a_o)                                
               rotation_2[:,2,1]     = np.sin(prop_plane_angle + rot*a_o)                                        
               rotation_2[:,2,2]     = np.cos(prop_plane_angle + rot*a_o)           
               rotation_2[:,3,3]     = 1
               
               inv_rotation_2        = np.zeros((dim_radial,4,4))
               inv_rotation_2[:,0,0] = 1                      
               inv_rotation_2[:,1,1] = np.cos(prop_plane_angle + rot*a_o)                                       
               inv_rotation_2[:,1,2] = np.sin(prop_plane_angle + rot*a_o)                                
               inv_rotation_2[:,2,1] = -np.sin(prop_plane_angle + rot*a_o) # -ve for inverse                                           
               inv_rotation_2[:,2,2] = np.cos(prop_plane_angle + rot*a_o)           
               inv_rotation_2[:,3,3] = 1
               
               # rotation about y axis by thust angle 
               rotation_3            = np.zeros((dim_radial,4,4))
               rotation_3[:,0,0]     =  np.cos(thrust_angle)      
               rotation_3[:,0,2]     =  np.sin(thrust_angle)             
               rotation_3[:,1,1]     =  1
               rotation_3[:,2,0]     = -np.sin(thrust_angle) # -ve for inverse
               rotation_3[:,2,2]     =  np.cos(thrust_angle)
               rotation_3[:,3,3]     = 1
               
               inv_rotation_3        = np.zeros((dim_radial,4,4))
               inv_rotation_3[:,0,0] =  np.cos(thrust_angle)      
               inv_rotation_3[:,0,2] = -np.sin(thrust_angle)             
               inv_rotation_3[:,1,1] =  1
               inv_rotation_3[:,2,0] = np.sin(thrust_angle)  
               inv_rotation_3[:,2,2] =  np.cos(thrust_angle)
               inv_rotation_3[:,3,3] = 1
               
               # translation to location on propeller on vehicle 
               translate_1            = np.repeat(np.atleast_3d(np.eye(4)).T,dim_radial,axis = 0) 
               translate_1[:,0,3]     = prop_origin[p_idx][0]   
               translate_1[:,1,3]     = prop_origin[p_idx][1]        
               translate_1[:,2,3]     = prop_origin[p_idx][2]   
               translate_1[:,3,3]     = 1                       
                                      
               inv_translate_1        = np.repeat(np.atleast_3d(np.eye(4)).T,dim_radial,axis = 0) 
               inv_translate_1[:,0,3] = -prop_origin[p_idx][0]   # -ve for inverse   
               inv_translate_1[:,1,3] = -prop_origin[p_idx][1]   # -ve for inverse         
               inv_translate_1[:,2,3] = -prop_origin[p_idx][2]   # -ve for inverse  
               inv_translate_1[:,3,3] = 1  
                
               # rotation about y axis by angle of attack
               rotation_4            = np.zeros((dim_radial,4,4))
               rotation_4[:,0,0]     =  np.cos(AoA)      
               rotation_4[:,0,2]     = -np.sin(AoA)             
               rotation_4[:,1,1]     =  1      
               rotation_4[:,2,0]     =  np.sin(AoA)  
               rotation_4[:,2,2]     =  np.cos(AoA)
               rotation_4[:,3,3]     = 1               
               
               inv_rotation_4        = np.zeros((dim_radial,4,4))
               inv_rotation_4[:,0,0] =  np.cos(AoA)      
               inv_rotation_4[:,0,2] =  np.sin(AoA)             
               inv_rotation_4[:,1,1] =  1      
               inv_rotation_4[:,2,0] = -np.sin(AoA) # -ve for inverse
               inv_rotation_4[:,2,2] =  np.cos(AoA)
               inv_rotation_4[:,3,3] = 1               
               
               # translation to location on propeller on vehicle 
               translate_2            = np.repeat(np.atleast_3d(np.eye(4)).T,dim_radial,axis = 0) 
               translate_2[:,0,3]     = mls[ctrl_pt,mic_loc,0]  
               translate_2[:,1,3]     = mls[ctrl_pt,mic_loc,1]       
               translate_2[:,2,3]     = mls[ctrl_pt,mic_loc,2]  
               
               inv_translate_2        = np.repeat(np.atleast_3d(np.eye(4)).T,dim_radial,axis = 0) 
               inv_translate_2[:,0,3] = - mls[ctrl_pt,mic_loc,0] # -ve for inverse 
               inv_translate_2[:,1,3] = - mls[ctrl_pt,mic_loc,1] # -ve for inverse      
               inv_translate_2[:,2,3] = - mls[ctrl_pt,mic_loc,2] # -ve for inverse  

               # propogation time  
               tau_r                 = S_r/a
          
               # create retarded coordinates  
               translate_2_r            = np.repeat(np.atleast_3d(np.eye(4)).T,dim_radial,axis = 0) 
               translate_2_r[:,0,3]     = mls[ctrl_pt,mic_loc,0] + Vx*tau_r  
               translate_2_r[:,1,3]     = mls[ctrl_pt,mic_loc,1] + Vy*tau_r       
               translate_2_r[:,2,3]     = mls[ctrl_pt,mic_loc,2] + Vz*tau_r   
          
               inv_translate_2_r        = np.repeat(np.atleast_3d(np.eye(4)).T,dim_radial,axis = 0) 
               inv_translate_2_r[:,0,3] = - mls[ctrl_pt,mic_loc,0] - Vx*tau_r # -ve for inverse 
               inv_translate_2_r[:,1,3] = - mls[ctrl_pt,mic_loc,1] - Vy*tau_r # -ve for inverse      
               inv_translate_2_r[:,2,3] = - mls[ctrl_pt,mic_loc,2] - Vz*tau_r # -ve for inverse  
               
               # ---------------------------------------------------------------------------------------------
               # ROTATE POINTS
               # ---------------------------------------------------------------------------------------------  
               # X matrix 
               # need to validate ( may be X')
               X_mat_1  =  np.matmul(inv_rotation_1_1,X_mat_0[...,None]).squeeze()      
               X_mat_2  =  np.matmul(inv_rotation_2  ,X_mat_1[...,None]).squeeze()
               X_mat_3  =  np.matmul(inv_rotation_3  ,X_mat_2[...,None]).squeeze()
               X_mat_4  =  np.matmul(inv_translate_1 ,X_mat_3[...,None]).squeeze()
               X_mat_5  =  np.matmul(inv_rotation_4  ,X_mat_4[...,None]).squeeze()
               X_mat_6  =  np.matmul(inv_translate_2 ,X_mat_5[...,None]).squeeze()
                                           
               # X' matrix         
               Xp_mat_0 =  X_mat_6
               Xp_mat_1 =  np.matmul(inv_rotation_1_1,Xp_mat_0[...,None]).squeeze()
               Xp_mat_2 =  np.matmul(inv_rotation_2  ,Xp_mat_1[...,None]).squeeze()
               Xp_mat_3 =  np.matmul(inv_rotation_3  ,Xp_mat_2[...,None]).squeeze()
               Xp_mat_4 =  np.matmul(inv_rotation_4  ,Xp_mat_3[...,None]).squeeze() 
               
               #Xp_mat_0 =  X_mat_6
               #Xp_mat_1 =  np.matmul(rotation_4  ,Xp_mat_0[...,None]).squeeze()
               #Xp_mat_2 =  np.matmul(rotation_3  ,Xp_mat_1[...,None]).squeeze()
               #Xp_mat_3 =  np.matmul(rotation_2  ,Xp_mat_2[...,None]).squeeze()
               #Xp_mat_4 =  np.matmul(rotation_1_1,Xp_mat_3[...,None]).squeeze()
               
               # Xe matrix          
               Xe_mat_1  =  np.matmul(inv_rotation_1_2,Xe_mat_0[...,None]).squeeze()   
               Xe_mat_2  =  np.matmul(inv_rotation_2  ,Xe_mat_1[...,None]).squeeze()
               Xe_mat_3  =  np.matmul(inv_rotation_3  ,Xe_mat_2[...,None]).squeeze()
               Xe_mat_4  =  np.matmul(inv_translate_1 ,Xe_mat_3[...,None]).squeeze()
               Xe_mat_5  =  np.matmul(inv_rotation_4  ,Xe_mat_4[...,None]).squeeze()
               Xe_mat_6  =  np.matmul(inv_translate_2 ,Xe_mat_5[...,None]).squeeze()  
          
               # Retarded Matrices 
               # Xr matrix   
               Xr_mat_1  =  np.matmul(inv_rotation_1_1 ,Xr_mat_0[...,None]).squeeze()  
               Xr_mat_2  =  np.matmul(inv_rotation_2   ,Xr_mat_1[...,None]).squeeze()
               Xr_mat_3  =  np.matmul(inv_rotation_3   ,Xr_mat_2[...,None]).squeeze()
               Xr_mat_4  =  np.matmul(inv_translate_1  ,Xr_mat_3[...,None]).squeeze()
               Xr_mat_5  =  np.matmul(inv_rotation_4   ,Xr_mat_4[...,None]).squeeze()
               Xr_mat_6  =  np.matmul(inv_translate_2_r,Xr_mat_5[...,None]).squeeze()
          
               # X'r matrix                 
               Xpr_mat_0 =  Xr_mat_6
               Xpr_mat_1 =  np.matmul(rotation_4,Xpr_mat_0[...,None]).squeeze()
               Xpr_mat_2 =  np.matmul(rotation_3,Xpr_mat_1[...,None]).squeeze()
               Xpr_mat_3 =  np.matmul(rotation_2,Xpr_mat_2[...,None]).squeeze()
               Xpr_mat_4 =  np.matmul(rotation_1_1,Xpr_mat_3[...,None]).squeeze()
          
               # Xer matrix     
               Xer_mat_1  =  np.matmul(inv_rotation_1_2,Xer_mat_0[...,None]).squeeze()   
               Xer_mat_2  =  np.matmul(inv_rotation_2  ,Xer_mat_1[...,None]).squeeze()
               Xer_mat_3  =  np.matmul(inv_rotation_3  ,Xer_mat_2[...,None]).squeeze()
               Xer_mat_4  =  np.matmul(inv_translate_1 ,Xer_mat_3[...,None]).squeeze()
               Xer_mat_5  =  np.matmul(inv_rotation_4  ,Xer_mat_4[...,None]).squeeze()
               Xer_mat_6  =  np.matmul(inv_translate_2_r ,Xer_mat_5[...,None]).squeeze()  
               
               
               # ---------------------------------------------------------------------------------------------
               # STORE POINTS
               # ---------------------------------------------------------------------------------------------                
               X_mat[B_idx,:,n_a,0]  = X_mat_6[:,0]
               X_mat[B_idx,:,n_a,1]  = X_mat_6[:,1]
               X_mat[B_idx,:,n_a,2]  = X_mat_6[:,2]
               
               Xp_mat[B_idx,:,n_a,0] = Xp_mat_4[:,0]
               Xp_mat[B_idx,:,n_a,1] = Xp_mat_4[:,1]
               Xp_mat[B_idx,:,n_a,2] = Xp_mat_4[:,2]
               
               Xe_mat[B_idx,:,n_a,0] = Xe_mat_6[:,0]
               Xe_mat[B_idx,:,n_a,1] = Xe_mat_6[:,1]
               Xe_mat[B_idx,:,n_a,2] = Xe_mat_6[:,2] 
               
               Xr_mat[B_idx,:,n_a,0]  = Xr_mat_6[:,0]
               Xr_mat[B_idx,:,n_a,1]  = Xr_mat_6[:,1]
               Xr_mat[B_idx,:,n_a,2]  = Xr_mat_6[:,2]
               
               Xpr_mat[B_idx,:,n_a,0] = Xpr_mat_4[:,0]
               Xpr_mat[B_idx,:,n_a,1] = Xpr_mat_4[:,1]
               Xpr_mat[B_idx,:,n_a,2] = Xpr_mat_4[:,2]
               
               Xer_mat[B_idx,:,n_a,0] = Xer_mat_6[:,0]
               Xer_mat[B_idx,:,n_a,1] = Xer_mat_6[:,1]
               Xer_mat[B_idx,:,n_a,2] = Xer_mat_6[:,2]     
                             
     return  X_mat, Xp_mat , Xe_mat, Xr_mat, Xpr_mat, Xer_mat , tau_r

#def compute_H_p(f,delta_star_p,U,M,Re_c,Re_delta_star_p):
     
     
     #return H_p


#def compute_H_s(f,delta_star_s,U,M,Re_c):
     
     
     #return H_s 

#def compute_H_alpha(f,delta_star_s,alpha,M,Re_c):
     
     
     
     #return H_alpha