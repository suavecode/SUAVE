# noise_propeller_low_fidelty.py
#
# Created:  Feb 2018, M. Clarke

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
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import SPL_spectra_arithmetic
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import senel_noise
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import dbA_noise
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import SPL_harmonic_to_third_octave
from SUAVE.Methods.Noise.Fidelity_One.Propeller.compute_broadband_noise   import compute_broadband_noise

## @ingroupMethods-Noise-Fidelity_One-Propeller
def propeller_low_fidelity(network,propeller,auc_opts,segment,settings, mic_loc, harmonic_test ):
    ''' This computes the sound pressure level of a system of rotating blades           
        - Hanson method used to compute rotational noise  
        - Vortex noise is computed using brooks & burley 
        
    Inputs:
        - noise_data	 - SUAVE type vehicle
        - Note that the notation is different from the reference, the speed of sound is denotes as a not c
        and airfoil thickness is denoted with "t" and not "h"
    
    Outputs:
        SPL     (using 3 methods*)   - Overall Sound Pressure Level, [dB] 
        SPL_dBA (using 3 methods*)   - Overall Sound Pressure Level, [dBA] 
    
    Properties Used:
        N/A   
    '''
    
    # unpack 
    conditions             = segment.state.conditions
    microphone_locations   = conditions.noise.microphone_locations
    angle_of_attack        = conditions.aerodynamics.angle_of_attack 
    velocity_vector        = conditions.frames.inertial.velocity_vector
    freestream             = conditions.freestream 
    N                      = network.number_of_engines                    
    ctrl_pts               = len(angle_of_attack) 
    num_f                  = len(settings.center_frequencies)
    
    # create empty arrays for results  
    SPL_tot                 = np.zeros(ctrl_pts) 
    SPL_dBA_tot             = np.zeros(ctrl_pts)  
    SPL_dBA_prop            = np.zeros((ctrl_pts,N))
    SPL_prop_bb_spectrum    = np.zeros((ctrl_pts,N,num_f))  
    SPL_prop_h_spectrum     = np.zeros((ctrl_pts,N,num_f))    
    SPL_prop_h_dBA_spectrum = np.zeros((ctrl_pts,N,num_f)) 
    SPL_prop_spectrum       = np.zeros((ctrl_pts,N,num_f))  
    SPL_prop_tonal_spectrum = np.zeros((ctrl_pts,N,num_f))  
    SPL_prop_bpfs_spectrum  = np.zeros((ctrl_pts,N,num_f))  
    
    
    SPL_tot_bb_spectrum    = np.zeros((ctrl_pts,num_f))    
    SPL_tot_spectrum       = np.zeros((ctrl_pts,num_f))  
    SPL_tot_tonal_spectrum = np.zeros((ctrl_pts,num_f))  
    SPL_tot_bpfs_spectrum  = np.zeros((ctrl_pts,num_f))
                                       
    if harmonic_test.any():  
        SPL  = np.zeros((ctrl_pts,len(harmonic_test)))
        SPL_v  = np.zeros_like(SPL) 

    # loop for control points  
    for i in range(ctrl_pts):            
        if harmonic_test.any():
            harmonics    = harmonic_test
        else:
            harmonics    = np.arange(1,20) 
            
        num_h = len(harmonics)
        
        SPL_r         = np.zeros((N,num_h))  
        SPL_r_dBA     = np.zeros_like(SPL_r)  
        p_pref_r      = np.zeros_like(SPL_r) 
        p_pref_r_dBA  = np.zeros_like(SPL_r)     
        f             = np.zeros(num_h)
        
        for p_idx in range(N):
            
            
            # ----------------------------------------------------------------------------------
            # Rotational Noise  Thickness and Loading Noise
            # ----------------------------------------------------------------------------------        
            for h in range(num_h):
                m              = harmonics[h]                                      # harmonic number 
                p_ref          = 2E-5                                              # referece atmospheric pressure
                a              = freestream.speed_of_sound[i][0]                   # speed of sound
                rho            = freestream.density[i][0]                          # air density 
                AoA            = angle_of_attack[i][0]                             # vehicle angle of attack  
                thrust_angle   = auc_opts.thrust_angle                             # propeller thrust angle
                alpha          = AoA + thrust_angle   
                x , y , z      = compute_coordinates(i,mic_loc,p_idx,AoA,thrust_angle,microphone_locations,propeller.origin)  
                Vx             = velocity_vector[i][0]                             # x velocity of propeller  
                Vy             = velocity_vector[i][1]                             # y velocity of propeller 
                Vz             = velocity_vector[i][2]                             # z velocity of propeller 
                B              = propeller.number_of_blades                        # number of propeller blades
                omega          = auc_opts.omega[i]                                 # angular velocity       
                dT_dr          = auc_opts.blade_dT_dr[i]                           # nondimensionalized differential thrust distribution 
                dQ_dr          = auc_opts.blade_dQ_dr[i]                           # nondimensionalized differential torque distribution
                R              = propeller.radius_distribution                     # radial location     
                c              = propeller.chord_distribution                      # blade chord    
                R_tip          = propeller.tip_radius
                beta           = propeller.twist_distribution                      # twist distribution  
                t              = propeller.max_thickness_distribution              # thickness distribution
                t_c            = propeller.thickness_to_chord                      # thickness to chord ratio
                MCA            = propeller.mid_chord_aligment                      # Mid Chord Alighment 
                                                                                  
                f[h]          = B*omega*m/(2*np.pi)   
                n             = len(R)  
                D             = 2*R[-1]                                            # propeller diameter    
                r             = R/R[-1]                                            # non dimensional radius distribution   
                S             = np.sqrt(x**2 + y**2 + z**2)                        # distance between rotor and the observer    
                theta         = np.arccos(x/S)                                     
                Y             = np.sqrt(y**2 + z**2)                               # observer distance from propeller axis          
                V             = np.sqrt(Vx**2 + Vy**2 + Vz**2)                     # velocity magnitude
                M_x           = V/a     
                V_tip         = R_tip*omega                                        # blade_tip_speed 
                M_t           = V_tip/a                                            # tip Mach number 
                M_r           = np.sqrt(M_x**2 + (r**2)*(M_t**2))                    # section relative Mach number     
                B_D           = c/D 
                phi           = np.arctan(z/y)                                     # tangential angle   
                
                theta_r       = np.arccos(np.cos(theta)*np.sqrt(1 - (M_x**2)*(np.sin(theta))**2) + M_x*(np.sin(theta))**2 )       # retarted  theta angle in the retarded reference frame
                theta_r_prime = np.arccos(np.cos(theta_r)*np.cos(alpha) + np.sin(theta_r)*np.sin(phi)*np.sin(alpha) )    
                phi_prime     = np.arccos((np.sin(theta_r)/np.sin(theta_r_prime))*np.cos(phi))                            # phi angle relative to propeller shaft axis                                                   
                k_x           = ((2*m*B*B_D*M_t)/(M_r*(1 - M_x*np.cos(theta_r))))      # wave number
                k_y           = ((2*m*B*B_D)/(M_r*r)) *((M_x - (M_r**2)*np.cos(theta_r))/(1 - M_x*np.cos(theta_r)))
                phi_s         = ((2*m*B*M_t)/(M_r*(1 - M_x*np.cos(theta_r))))*(MCA/D)
                S_r           = Y/(np.sin(theta_r))                                # distance in retarded reference frame   
                Jmb           = jv(m*B,((m*B*r*M_t*np.sin(theta_r_prime))/(1 - M_x*np.cos(theta_r))))  
                psi_L         = np.zeros(n)
                psi_V         = np.zeros(n)
                
                for idx in range(n):
                    if k_x[idx] == 0:                        
                        psi_V[idx] = 2/3                                           # normalized thickness souce transforms
                        psi_L[idx] = 1                                             # normalized loading souce transforms
                    else:  
                        psi_V[idx] = (8/(k_x[idx]**2))*((2/k_x[idx])*np.sin(0.5*k_x[idx]) - np.cos(0.5*k_x[idx]))                    # normalized thickness souce transforms           
                        psi_L[idx] = (2/k_x[idx])*np.sin(0.5*k_x[idx])             # normalized loading souce transforms             
                
                # ---------------------------------------------------------------------------------  
                # Aeroacoustics of Flight Vehicles Theory and Practice
                # ---------------------------------------------------------------------------------  
                # sound pressure for thickness noise  
            
                #exponent_fraction = np.exp(1j*m*B*((omega*S_r/a)  - np.pi/2))/(1 - M_x*np.cos(theta_r))            
                #p_mT_H_function = ((rho*(a**2)*B*np.sin(theta_r))/(8*np.pi*(Y/D)))* exponent_fraction
                #p_mT_H_integral = np.trapz(((M_r**2)*(t_c)*np.exp(1j*phi_s)*Jmb*(k_x**2)*psi_V ),x = r)
                #p_mT_H = -p_mT_H_function*p_mT_H_integral/np.sqrt(2) 
                #p_mT_H = abs(p_mT_H)    
                
                ## sound pressure for loading noise 
                #p_mL_H_function  = (1j*m*B*M_t*np.sin(theta_r))/(4*np.pi*Y*R_tip*(1 - M_x*np.cos(theta_r))) 
                #p_mL_H_integral  = np.trapz((((np.cos(theta_r_prime)/(1 - M_x*np.cos(theta_r)))*dT_dr - (1/((r**2)*M_t*R_tip))*dQ_dr)
                                                     #* np.exp(1j*phi_s)*Jmb * psi_L),x = r)
                #p_mL_H =  p_mL_H_function*p_mL_H_integral/np.sqrt(2) 
                #p_mL_H =  abs(p_mL_H) 
                            
                # ---------------------------------------------------------------------------------        
                # Applicability of Early Acoustic Theory for Modern Propeller Design
                # ---------------------------------------------------------------------------------     
                
                # sound pressure for thickness noise  
                exponent_fraction = np.exp(1j*m*B*((omega*S_r/a) +  phi_prime - np.pi/2))/(1 - M_x*np.cos(theta_r))
                p_mT_H_function   = ((rho*(a**2)*B*np.sin(theta_r))/(4*np.sqrt(2)*np.pi*(Y/D)))* exponent_fraction
                p_mT_H_integral   = np.trapz(((M_r**2)*(t_c)*np.exp(1j*phi_s)*Jmb*(k_x**2)*psi_V ),x = r)
                p_mT_H            = -p_mT_H_function*p_mT_H_integral   
                p_mT_H            = abs(p_mT_H)             
                
                # sound pressure for loading noise 
                p_mL_H_function   = (m*B*M_t*np.sin(theta_r)/ (2*np.sqrt(2)*np.pi*Y*R_tip)) *exponent_fraction
                p_mL_H_integral   = np.trapz((((np.cos(theta_r_prime)/(1 - M_x*np.cos(theta_r)))*dT_dr - (1/((r**2)*M_t*R_tip))*dQ_dr)
                                             * np.exp(1j*phi_s)*Jmb * psi_L),x = r)
                p_mL_H            =  p_mL_H_function*p_mL_H_integral 
                p_mL_H            =  abs(p_mL_H) 
            
                
                # unweighted rotational sound pressure level 
                SPL_r[p_idx,h]        = 20*np.log10((np.linalg.norm(p_mL_H + p_mT_H))/p_ref) 
                p_pref_r[p_idx,h]     = 10**(SPL_r[p_idx,h]/10)   
                SPL_r_dBA[p_idx,h]    = A_weighting(SPL_r[p_idx,h],f[h]) 
                p_pref_r_dBA[p_idx,h] = 10**(SPL_r_dBA[p_idx,h]/10)    
             
            # convert to 1/3 octave spectrum 
            SPL_prop_h_spectrum[i,p_idx,:]      = SPL_harmonic_to_third_octave(SPL_r[p_idx,:],f,settings)         
            SPL_prop_h_dBA_spectrum[i,p_idx,:]  = SPL_harmonic_to_third_octave(SPL_r_dBA[p_idx,:],f,settings)     
            
            # ------------------------------------------------------------------------------------
            # Broadband Noise (Vortex Noise)   
            # ------------------------------------------------------------------------------------ 
            G_BB      = compute_broadband_noise(i, p_idx, mic_loc,propeller,auc_opts,segment,S_r)       
            p_pref_bb_dBA        = 0
            SPL_prop_bb_spectrum = 0
            #V_07      = V_tip*0.70/(Units.feet)                                      # blade velocity at r/R_tip = 0.7 
            #St        = 0.28                                                         # Strouhal number             
            #t_avg     = np.mean(t)/(Units.feet)                                      # thickness
            #c_avg     = np.mean(c)/(Units.feet)                                      # average chord  
            #beta_07   = beta[round(n*0.70)]                                          # blade angle of attack at r/R = 0.7
            #h_val     = t_avg*np.cos(beta_07) + c_avg*np.sin(beta_07)                # projected blade thickness                   
            #f_peak    = (V_07*St)/h_val                                              # V - blade velocity at a radial location of 0.7              
            #A_blade   = (np.trapz(c, x = R))/(Units.feet**2) # area of blade      
            #CL_07     = 2*np.pi*beta_07
            #S_feet    = S/(Units.feet)
            #SPL_300ft = 10*np.log10(((6.1E-27)*A_blade*V_07**6)/(10**-16)) + 20*np.log(CL_07/0.4)  
            #SPL_v     = SPL_300ft - 20*np.log10(S_feet/300)                          # CORRECT 
            
            ## estimation of A-Weighting for Vortex Noise  
            #f_v         = np.array([0.5*f_peak[0],1*f_peak[0],2*f_peak[0],4*f_peak[0],8*f_peak[0],16*f_peak[0]]) # spectrum
            #fr          = f_v/f_peak                                          # frequency ratio  
            #SPL_weight  = [7.92 , 4.17 , 8.33 , 8.75 ,12.92 , 13.33]                 # SPL weight
            #SPL_v       = np.ones_like(SPL_weight)*SPL_v - SPL_weight                # SPL correction
            
            #dim          = len(f_v)
            #C            = np.zeros(dim) 
            #p_pref_bb_dBA = np.zeros(dim-1)
            #SPL_bb_dbAi   = np.zeros(dim)
            
            #for j in range(dim):
                #SPL_bb_dbAi[j] = A_weighting(SPL_v[j],f_v[j])
            
            #for j in range(dim-1):
                #C[j]            = (SPL_bb_dbAi[j+1] - SPL_bb_dbAi[j])/(np.log10(fr[j+1]) - np.log10(fr[j])) 
                #C[j+1]          = SPL_bb_dbAi[j+1] - C[j]*np.log10(fr[j+1])   
                #p_pref_bb_dBA[j] = (10**(0.1*C[j+1]))* (  ((fr[j+1]**(0.1*C[j] + 1 ))/(0.1*C[j] + 1 )) - ((fr[j]**(0.1*C[j] + 1 ))/(0.1*C[j] + 1 )) ) 
            
            #p_pref_bb_dBA[np.isnan(p_pref_bb_dBA)] = 0
            
            # ---------------------------------------------------------------------------
            # convert to 1/3 octave spectrum  
            # ---------------------------------------------------------------------------
            #SPL_prop_bb_spectrum[i,p_idx,:]     = SPL_harmonic_to_third_octave(SPL_v,f_v,settings)
            
            # ---------------------------------------------------------------------------
            # Rotational(periodic/tonal)  
            # --------------------------------------------------------------------------- 
            SPL_prop_tonal_spectrum[i,p_idx,:]           = 10*np.log10( 10**(SPL_prop_h_spectrum[i,p_idx,:]/10))
            
            # ---------------------------------------------------------------------------
            # Rotational(periodic/tonal) and Vortex (broadband)
            # --------------------------------------------------------------------------- 
            SPL_prop_bpfs_spectrum[i,p_idx,:num_h] = SPL_r 
            SPL_prop_spectrum[i,p_idx,:]           = 10*np.log10( 10**(SPL_prop_h_spectrum[i,p_idx,:]/10) +  10**(SPL_prop_bb_spectrum[i,p_idx,:]/10) )
            
            # pressure ratios used to combine A weighted sound since decibel arithmetic does not work for broadband noise since it is a continuous spectrum 
            total_p_pref_dBA = np.concatenate([p_pref_r_dBA[p_idx,:],p_pref_bb_dBA])
            SPL_dBA_prop[i,p_idx]   = pressure_ratio_to_SPL_arithmetic(total_p_pref_dBA)  
            SPL_dBA_prop[np.isinf(SPL_dBA_prop)] = 0 
        
        
        # Summation of spectra from propellers into into one SPL
        SPL_tot[i]                  =  SPL_arithmetic(SPL_prop_spectrum[i]) 
        SPL_dBA_tot[i]              =  SPL_arithmetic(SPL_dBA_prop[i])      
        SPL_tot_spectrum[i,:]       =  SPL_spectra_arithmetic(SPL_prop_spectrum[i])        
        SPL_tot_bpfs_spectrum[i,:]  =  SPL_spectra_arithmetic(SPL_prop_bpfs_spectrum[i])           
        SPL_tot_tonal_spectrum[i,:] =  SPL_spectra_arithmetic(SPL_prop_tonal_spectrum[i]) 
        SPL_tot_bb_spectrum[i,:]    =  SPL_spectra_arithmetic(SPL_prop_bb_spectrum[i])    
    
    # Pack Results
    propeller_noise                   = Data() 
    
    # total sound pressure level 
    propeller_noise.SPL               = SPL_tot
    propeller_noise.SPL_dBA           = SPL_dBA_tot   
    
    # sound spectra 
    propeller_noise.SPL_spectrum      = SPL_tot_spectrum      # 1/3 octave band 
    propeller_noise.SPL_bpfs_spectrum = SPL_tot_bpfs_spectrum # blade passing frequency specturm (only rotational noise)
    propeller_noise.SPL_tonal         = SPL_tot_tonal_spectrum
    propeller_noise.SPL_broadband     = SPL_tot_bb_spectrum
    
    return propeller_noise



def compute_coordinates(i,mic_loc,p_idx,AoA,thrust_angle,mls,prop_origin):  
    
    # rotation of propeller about y axis by thurst angle (one extra dimension for translations)
    rotation_1      = np.zeros((4,4))
    rotation_1[0,0] = np.cos(thrust_angle)           
    rotation_1[0,2] = np.sin(thrust_angle)                 
    rotation_1[1,1] = 1
    rotation_1[2,0] = -np.sin(thrust_angle) 
    rotation_1[2,2] = np.cos(thrust_angle)      
    rotation_1[3,3] = 1     
    
    # translation to location on propeller
    translation_1      = np.eye(4)
    translation_1[0,3] = prop_origin[p_idx][0]     
    translation_1[1,3] = prop_origin[p_idx][1]           
    translation_1[2,3] = prop_origin[p_idx][2] 
    
    # rotation of vehicle about y axis by AoA 
    rotation_2      = np.zeros((4,4))
    rotation_2[0,0] = np.cos(AoA)           
    rotation_2[0,2] = np.sin(AoA)                 
    rotation_2[1,1] = 1
    rotation_2[2,0] = -np.sin(AoA) 
    rotation_2[2,2] = np.cos(AoA)     
    rotation_2[3,3] = 1 
    
    # translation of vehicle to air 
    translate_2      = np.eye(4)
    translate_2[0,3] = mls[i,mic_loc,0]  
    translate_2[1,3] = mls[i,mic_loc,1]   
    translate_2[2,3] = mls[i,mic_loc,2] 
    
    mat_0  = np.array([[0],[0],[0],[1]])
    
    # execute operation  
    mat_1 = np.matmul(rotation_1,mat_0) 
    mat_2 = np.matmul(translation_1,mat_1)
    mat_3 = np.matmul(rotation_2,mat_2) 
    mat_4 = np.matmul(translate_2,mat_3)
    mat_4 = -mat_4
    
    x = mat_4[0,0] 
    y = mat_4[1,0] 
    z = mat_4[2,0] 

    return x , y , z