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
def compute_broadband_noise(i, mic_loc,network,propeller,auc_opts,segment,settings ):
     
     conditions             = segment.state.conditions
     microphone_location    = conditions.noise.microphone_locations
     angle_of_attack        = conditions.aerodynamics.angle_of_attack 
     velocity_vector        = conditions.frames.inertial.velocity_vector
     freestream             = conditions.freestream     
     
     
     a              = freestream.speed_of_sound[i][0]                   # speed of sound
     rho            = freestream.density[i][0]                          # air density 
     x              = microphone_location[i,mic_loc,0] # + propeller.origin[0][0]  # x relative position from observer
     y              = microphone_location[i,mic_loc,1] # + propeller.origin[0][1]  # y relative position from observer 
     z              = microphone_location[i,mic_loc,2] # + propeller.origin[0][2]  # z relative position from observer
     Vx             = velocity_vector[i][0]                             # x velocity of propeller  
     Vy             = velocity_vector[i][1]                             # y velocity of propeller 
     Vz             = velocity_vector[i][2]                             # z velocity of propeller 
     thrust_angle   = auc_opts.thrust_angle                             # propeller thrust angle
     AoA            = angle_of_attack[i][0]                             # vehicle angle of attack                                            
     N              = network.number_of_engines                         # numner of propeller
     B              = propeller.number_of_blades                        # number of propeller blades
     omega          = auc_opts.omega[i]                                 # angular velocity         
     r              = propeller.radius_distribution                     # radial location     
     r_2d           = # get from propeller.py
     
     
     c              = propeller.chord_distribution                      # blade chord    
     R_tip          = propeller.tip_radius
     beta           = propeller.twist_distribution                      # twist distribution  
     t              = propeller.max_thickness_distribution              # thickness distribution
     t_c            = propeller.thickness_to_chord                      # thickness to chord ratio
     MCA            = propeller.mid_chord_aligment                      # Mid Chord Alighment
     
     n             = len(L)     
     S             = np.sqrt(x**2 + y**2 + z**2)                        # distance between rotor and the observer    
     theta         = np.arccos(x/S)                                     
     alpha         = AoA + thrust_angle    
     Y             = np.sqrt(y**2 + z**2)                               # observer distance from propeller axis          
     V             = np.sqrt(Vx**2 + Vy**2 + Vz**2)                     # velocity magnitude
     M_x           = V/a     
     V_tip         = R_tip*omega                                        # blade_tip_speed 
     M_t           = V_tip/a                                            # tip Mach number 
     M_r           = np.sqrt(M_x**2 + (r**2)*(M_t**2))                    # section relative Mach number      
     
     V_tot         = # get from propeller.py
     
     # coordinate system transformation
     # vector from observer to vehicle 
     
     # vector from vehicle to rotor/propeller
     
     # vector from propeller to blade segment oriented by thurst angle 
     
     
     # trailing edge vector 
     
     
     # total vector
     
     Xe            =
     
     
     X_r_prime 
     cos_zeta      = np.dot(X_r_prime,V_tot)/(np.linalg.norm(X_r_prime)*V_tot)
     l             =  # spanwize extent of the vortex formation at the trailing edge 
     alpha_eff_tip = 
     M_max         = alpha_eff_tip , M   # maximum velocity in or about the vortex tear the trailing edge alpha_eff_tip and M #
     phi           = 14*Units.degrees # solid angle between the bkade surfaces immediarely upstream of the trailing edge  # ref 27
     h             = 9E-3 # trailing edge thickness 
     
     delta_star_avg = # average boundary layer displacement thickness of the pressure and suction sides any tailing edge 
     U              = # flow normal to the span (2 D array from hub to tip and azimuth) 
     delta_star_p   = # displacement thickness pressure side   # ref 27 
     delta_star_s   = # dispacement thickness suction side  # ref 27  
     L              = r[1:] - r[:-1] # blade segment width spanwise
     M              = U/a   
     D_bar_h        =  
     r_e            = S     # observer distance form trailing edge
     
     r_er     = np.linalg.norm(X_er) # or np.sqrt(z_er**2 +  y_er**2 + z_er**2)
     Theta_er = np.arccos(x_er/r_er)
     Phi_er   = np.arccos(y_er/np.sqrt(y_er**2 + z_er**2)) 
     
     D_bar_l = ((np.sin(Theta_er/2)**2)*np.sin(Phi_er)**2)/(1 - M_tot*cos_zeta)**4
     
     D_bar_h =  (2*(np.sin(Theta_er/2)**2)*np.sin(Phi_er)**2)/(1 - M_tot*cos_zeta)**4
     
     U = omega*r_2d
     # that the boundary layer displacement thicknesses on the two sides, as well as boundary layer thicknesses
     # to be used below, are computed as functions27 of Rec and a* which were derived from detailed 
     # near-wake flow measurements from the aforementioned isolated NACA 0012 blade section studies
     for b in range(B)
         for m in range():
              for n in range(stop):
                  alpha_star = 
                  if np.norm(alpha_star) < 12.5*Units.degrees or   : 
                       # Turbulent Boundary Layer - Trailing Edge noise
                       ##  pressure side 
                       G_TBL_TEp      = 0
                       
                       ## suction side
                       G_TBL_TEs      = 0
                       
                       ## noise at angle of attack 
                       H_alpha        = compute_H_alpha(f,delta_star_s,alpha,M,Re_c) 
                       T_TBL_TEalpha  = ((delta_star_s*(M**5)*L*D_bar_l)/r_e**2)* H_alpha            
                  else:
                       # Turbulent Boundary Layer - Trailing Edge noise
                       ##  pressure side
                       H_p            = compute_H_p(f,delta_star_p,U,M,Re_c,Re_delta_star_p)  # ref 27  
                       G_TBL_TEp      = ((delta_star_p*(M**5)*L*D_bar_h)/r_e**2)* H_p 
                       
                       ## suction side
                       H_s            = compute_H_s(f,delta_star_s,U,M,Re_c)# ref 27  
                       G_TBL_TEs      = ((delta_star_p*(M**5)*L*D_bar_h)/r_e**2)* H_s  
                       
                       ## noise at angle of attack 
                       H_alpha        = compute_H_alpha(f,delta_star_s,alpha,M,Re_c)  # ref 27  
                       T_TBL_TEalpha  = ((delta_star_s*(M**5)*L*D_bar_h)/r_e**2)* H_alpha  
                       
                  # summation of Turbulent Boundary Layer - Trailing Edge noise sources 
                  G_TBL_TE = G_TBL_TEp + G_TBL_TEs + T_TBL_TEalpha  
                  G_TBL_TE_cor =  G_TBL_TE/(beta**2) # glauret correction 
                  # Laminar-boundary layer vortex shedding noise 
                  H_l      =  
                  G_LBL_VS = ((delta_p*(M**5)*L*D_bar_h)/r_e**2)* H_l  
                  
                  # Blunt Trailing Edge Noise 
                  H_b     =  
                  G_BTE   = ((h*(M**5.5)*L*D_bar_h)/r_e**2)* H_b  
                  
                  # Tip noise 
                  H_t    = 
                  G_Tip  = (((M**2)*(M_max**3)*l*D_bar_h)/r_e**2)* H_t 
                  
                  # Addition of noise sources 
                  G_self = G_TBL_TE + G_LBL_VS + G_BTE + G_Tip
                  
                  # Blade Wake Interaction 
                  M_tot    = np.linalg.norm(V_tot)/a
                  
                  sigma**2 = (np.linalg.norm(X_r)**2)*(1 - M_tot*np.cos(zeta))**2
                  zmu      = (zeta*omega)/U
                  or_as    = (omega*y_r)/(a*sigma**2)
                  G_BWI    = (((omega*z_r)/(4*np.pi*a*sigma**2))**2)*G_LE_delta_p*((b*C)**2)*(2*L*zmu)/(zmu**2 + or_as**2) 
                  
              
         G_BB     = np.sum(np.sum(4*(delta_phi/(2*np.pi))*(f_f0)(G_BWI + G_self)))
         
     G_BB_tot = 
     
     return G_BB_tot

def compute_H_p(f,delta_star_p,U,M,Re_c,Re_delta_star_p):
     
     
     return H_p


def compute_H_s(f,delta_star_s,U,M,Re_c):
     
     
     return H_s 

def compute_H_alpha(f,delta_star_s,alpha,M,Re_c) :
     
     
     
     return H_alpha