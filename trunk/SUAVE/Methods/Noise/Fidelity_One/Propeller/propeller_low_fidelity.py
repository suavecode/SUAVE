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
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import senel_noise
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import dbA_noise
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import SPL_harmonic_to_third_octave

## @ingroupMethods-Noise-Fidelity_One-Propeller
def propeller_low_fidelity(network,propeller,auc_opts,segment,settings, mic_loc, harmonic_test ):
    ''' This computes kal based procedure.           
        - Hanson method used to compute rotational noise  
        - Vortex noise is computed using the method outlined by Schlegel et. al 
        
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
    conditions           = segment.state.conditions
    microphone_location  = conditions.noise.microphone_locations
    angle_of_attack      = conditions.aerodynamics.angle_of_attack 
    velocity_vector      = conditions.frames.inertial.velocity_vector
    freestream           = conditions.freestream
    ctrl_pts             = len(angle_of_attack) 
    num_f                = len(settings.center_frequencies)
    
    # create empty arrays for results  
    SPL                   = np.zeros(ctrl_pts)
    SPL_v                 = np.zeros_like(SPL)   
    SPL_dBA_tot           = np.zeros_like(SPL)
    SPL_v_spectrum        = np.zeros((ctrl_pts,num_f))  
    SPL_h_spectrum        = np.zeros((ctrl_pts,num_f)) 
    SPL_h_dBA_spectrum    = np.zeros((ctrl_pts,num_f))   
    SPL_tot_spectrum      = np.zeros((ctrl_pts,num_f))  
    SPL_tot_bpfs_spectrum = np.zeros((ctrl_pts,num_f))
                                       
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
          
        SPL_r         = np.zeros(num_h)  
        SPL_r_dBA     = np.zeros_like(SPL_r)   
        p_pref_r      = np.zeros_like(SPL_r)  
        p_pref_r_dBA  = np.zeros_like(SPL_r)      
        f             = np.zeros_like(SPL_r)

        # ----------------------------------------------------------------------------------
        # Rotational Noise  Thickness and Loading Noise
        # ----------------------------------------------------------------------------------        
        for h in range(num_h):
            m              = harmonics[h]                                      # harmonic number 
            p_ref          = 2E-5                                              # referece atmospheric pressure
            a              = freestream.speed_of_sound[i][0]                   # speed of sound
            rho            = freestream.density[i][0]                          # air density 
            x              = microphone_location[i,mic_loc,0]                  # x relative position from observer
            y              = microphone_location[i,mic_loc,1]                  # y relative position from observer 
            z              = microphone_location[i,mic_loc,2]                  # z relative position from observer
            Vx             = velocity_vector[i][0]                             # x velocity of propeller  
            Vy             = velocity_vector[i][1]                             # y velocity of propeller 
            Vz             = velocity_vector[i][2]                             # z velocity of propeller 
            thrust_angle   = auc_opts.thrust_angle                             # propeller thrust angle
            AoA            = angle_of_attack[i][0]                             # vehicle angle of attack                                            
            N              = network.number_of_engines                         # numner of propeller
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
            D             = 2*R_tip                                            # propeller diameter    
            r             = R/R_tip                                            # non dimensional radius distribution   
            S             = np.sqrt(x**2 + y**2 + z**2)                        # distance between rotor and the observer    
            theta         = np.arccos(x/S)                                     
            alpha         = AoA + thrust_angle                                 
            Y             = np.sqrt(y**2 + z**2)                               # observer distance from propeller axis          
            V             = np.sqrt(Vx**2 + Vy**2 + Vz**2)                     # velocity magnitude
            M             = V/a                                                # Mach number  
            V_tip         = R_tip*omega                                        # blade_tip_speed 
            M_t           = V_tip/a                                            # tip Mach number 
            M_s           = np.sqrt(M**2 + (r**2)*(M_t**2))                    # section relative Mach number 
            r_t           = R_tip                                              # propeller tip radius
            phi           = np.arctan(z/y)                                     # tangential angle  
            theta_r       = np.arccos(np.cos(theta)*np.sqrt(1 - (M**2)*\
                            (np.sin(theta))**2) + M*(np.sin(theta))**2 )       # theta angle in the retarded reference frame
            theta_r_prime = np.arccos(np.cos(theta_r)*np.cos(alpha) + \
                            np.sin(theta_r)*np.sin(phi)*np.sin(alpha) )    
            phi_prime     = np.arccos((np.sin(theta_r)/np.sin(theta_r_prime)) 
                                      *np.cos(phi))                            # phi angle relative to propeller shaft axis                                                   
            phi_s         = ((2*m*B*M_t)/(M_s*(1 - M*np.cos(theta_r))))*(MCA/D)  # phase lag due to sweep         
            S_r           = Y/(np.sin(theta_r))                                # distance in retarded reference frame 
            k_x           = ((2*m*B*c*M_t)/(M_s*(1 - M*np.cos(theta_r))))      # wave number
            Jmb           = jv(m*B,((m*B*r*M_t*np.sin(theta_r_prime))/(1 - M*np.cos(theta_r))))
            psi_L         = np.zeros(n)
            psi_V         = np.zeros(n)
            
            for idx in range(n):
                if k_x[idx] == 0:                        
                    psi_V[idx] = 2/3                                           # normalized thickness souce transforms
                    psi_L[idx] = 1                                             # normalized loading souce transforms
                else:  
                    psi_V[idx] = (8/(k_x[idx]**2))*((2/k_x[idx])*np.sin(0.5*k_x[idx]) - np.cos(0.5*k_x[idx]))                    # normalized thickness souce transforms           
                    psi_L[idx] = (2/k_x[idx])*np.sin(0.5*k_x[idx])             # normalized loading souce transforms
                    
            # sound pressure for loading noise 
            exponent_fraction = np.exp(1j*m*B*((omega*S_r/a)+(phi_prime - np.pi/2)))/(1 - M*np.cos(theta_r)) 
            p_mL_H_function  = (m*B*M_t*np.sin(theta_r)/ (2*np.sqrt(2)*np.pi*Y*r_t)) *exponent_fraction
            p_mL_H_integral  = np.trapz((((np.cos(theta_r_prime)/(1 - M*np.cos(theta_r)))*(dT_dr) - (1/((r**2)*M_t*r_t))*(dQ_dr))
                                         * np.exp(1j*phi_s)*Jmb * psi_L),x = r)
            p_mL_H =  p_mL_H_function*p_mL_H_integral 
            p_mL_H =  abs(p_mL_H)
            
            # sound pressure for thickness noise  
            p_mT_H_function = ((rho*(a**2)*B*np.sin(theta_r))/(4*np.sqrt(2)*np.pi*(Y/D)))* exponent_fraction
            p_mT_H_integral = np.trapz(((M_s**2)*(t_c)*np.exp(1j*phi_s)*Jmb*(k_x**2)*psi_V ),x = r)
            p_mT_H = -p_mT_H_function*p_mT_H_integral   
            p_mT_H = abs(p_mT_H)
            
            # unweighted rotational sound pressure level 
            SPL_r[h]        = 20*np.log10(N*(np.linalg.norm(p_mL_H + p_mT_H))/p_ref)
            p_pref_r[h]     = 10**(SPL_r[h]/10)   
            SPL_r_dBA[h]    = A_weighting(SPL_r[h],f[h]) 
            p_pref_r_dBA[h] = 10**(SPL_r_dBA[h]/10)    
         
        # convert to 1/3 octave spectrum 
        SPL_h_spectrum[i,:]      = SPL_harmonic_to_third_octave(SPL_r,f,settings)         
        SPL_h_dBA_spectrum[i,:]  = SPL_harmonic_to_third_octave(SPL_r_dBA,f,settings)     
        
        # ------------------------------------------------------------------------------------
        # Broadband Noise (Vortex Noise)   
        # ------------------------------------------------------------------------------------ 
        V_07      = V_tip*0.70/(Units.feet)                                      # blade velocity at r/R_tip = 0.7 
        St        = 0.28                                                         # Strouhal number             
        t_avg     = np.mean(t)/(Units.feet)                                      # thickness
        c_avg     = np.mean(c)/(Units.feet)                                      # average chord  
        beta_07   = beta[round(n*0.70)]                                          # blade angle of attack at r/R = 0.7
        h_val     = t_avg*np.cos(beta_07) + c_avg*np.sin(beta_07)                # projected blade thickness                   
        f_peak    = (V_07*St)/h_val                                              # V - blade velocity at a radial location of 0.7              
        A_blade   = (np.trapz(c, x = R))/(Units.feet**2) # area of blade      
        CL_07     = 2*np.pi*beta_07
        S_feet    = S/(Units.feet)
        SPL_300ft = 10*np.log10(((6.1E-27)*A_blade*V_07**6)/(10**-16)) + 20*np.log(CL_07/0.4)  
        SPL_v     = SPL_300ft - 20*np.log10(S_feet/300)                          # CORRECT 
        
        # estimation of A-Weighting for Vortex Noise  
        f_v         = np.array([0.5*f_peak[0],1*f_peak[0],2*f_peak[0],4*f_peak[0],8*f_peak[0],16*f_peak[0]]) # spectrum
        fr          = f_v/f_peak                                          # frequency ratio  
        SPL_weight  = [7.92 , 4.17 , 8.33 , 8.75 ,12.92 , 13.33]                 # SPL weight
        SPL_v       = np.ones_like(SPL_weight)*SPL_v - SPL_weight                # SPL correction
        
        dim         = len(f_v)
        C           = np.zeros(dim) 
        p_pref_v_dBA= np.zeros(dim-1)
        SPL_v_dbAi  = np.zeros(dim)
        
        for j in range(dim):
            SPL_v_dbAi[j] = A_weighting(SPL_v[j],f_v[j])
        
        for j in range(dim-1):
            C[j]            = (SPL_v_dbAi[j+1] - SPL_v_dbAi[j])/(np.log10(fr[j+1]) - np.log10(fr[j])) 
            C[j+1]          = SPL_v_dbAi[j+1] - C[j]*np.log10(fr[j+1])   
            p_pref_v_dBA[j] = (10**(0.1*C[j+1]))* (  ((fr[j+1]**(0.1*C[j] + 1 ))/(0.1*C[j] + 1 )) - ((fr[j]**(0.1*C[j] + 1 ))/(0.1*C[j] + 1 )) ) 
        
        p_pref_v_dBA[np.isnan(p_pref_v_dBA)] = 0
        
        # ---------------------------------------------------------------------------
        # convert to 1/3 octave spectrum  
        # ---------------------------------------------------------------------------
        SPL_v_spectrum[i,:]     = SPL_harmonic_to_third_octave(SPL_v,f_v,settings)
        
        # ---------------------------------------------------------------------------
        # Combining Rotational(periodic) and Vortex (broadband)
        # --------------------------------------------------------------------------- 
        SPL_tot_bpfs_spectrum[i,:num_h] = SPL_r 
        SPL_tot_spectrum[i,:]           = 10*np.log10( 10**(SPL_h_spectrum[i,:]/10) +  10**(SPL_v_spectrum[i,:]/10) )
        
        # pressure ratios used to combine A weighted sound since decibel arithmetic does not work for broadband noise since it is a continuous spectrum 
        total_p_pref_dBA = np.concatenate([p_pref_r_dBA,p_pref_v_dBA])
        SPL_dBA_tot[i]   = pressure_ratio_to_SPL_arithmetic(total_p_pref_dBA)  
        SPL_dBA_tot[np.isinf(SPL_dBA_tot)] = 0 
        
    # Summation of spectrum into one SPL
    SPL_tot      =  SPL_arithmetic(SPL_tot_spectrum) 
    
    # Pack Results
    propeller_noise                   = Data() 
    propeller_noise.SPL               = SPL_tot
    propeller_noise.SPL_spectrum      = SPL_tot_spectrum     # 1/3 octave band 
    propeller_noise.SPL_bpfs_spectrum = SPL_tot_bpfs_spectrum # blade passing frequency specturm (only rotational noise)
    propeller_noise.SPL_dBA           = SPL_dBA_tot   
    
    return propeller_noise
