# noise_propeller_low_fidelty.py
#
# Created:  Feb 2018, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units
import numpy as np
from scipy.special import jv 

from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.decibel_arithmetic import pressure_ratio_to_SPL_arithmetic
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.dbA_noise          import A_weighting

## @ingroupMethods-Noise-Fidelity_One-Propeller
def propeller_low_fidelity(conditions,harmonic_test):
    ''' This computes the SPL of rotors and propellers using Frequency based methods'
    
    Source:
        1. Herniczek, M., Feszty, D., Meslioui, S., Park, JongApplicability of Early Acoustic Theory for Modern Propeller Design
        2. Schlegel, R., King, R., and Muli, H., Helicopter Rotor Noise Generation and Propagation, Technical Report, 
        US Army Aviation Material Laboratories, Fort Eustis, VA, 1966
  
    Assumptions:
        - Empirical based procedure.           
        - The three methods used to compute rotational noise SPL are 1) Gutin and Deming, 2) Barry and Magliozzi and 3) Hanson
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
    microphone_location  = conditions.noise.microphone_locations
    angle_of_attack      = conditions.aerodynamics.angle_of_attack 
    velocity_vector      = conditions.frames.inertial.velocity_vector
    freestream           = conditions.freestream
    ctrl_pts             = len(angle_of_attack)
    num_mic              = conditions.noise.number_of_microphones
    
    # create empty arrays for results 
    SPL_GD_unweighted      = np.zeros((ctrl_pts,num_mic))
    SPL_BM_unweighted      = np.zeros_like(SPL_GD_unweighted)
    SPL_H_unweighted       = np.zeros_like(SPL_GD_unweighted)
    SPL_v_unweighted       = np.zeros_like(SPL_GD_unweighted) 
    SPL_GDv_dBA            = np.zeros_like(SPL_GD_unweighted)
    SPL_BMv_dBA            = np.zeros_like(SPL_GD_unweighted)
    SPL_Hv_dBA             = np.zeros_like(SPL_GD_unweighted)    
    SPL_v_dBA              = np.zeros_like(SPL_GD_unweighted)
    
    if harmonic_test.any(): 
        SPL_GD_unweighted = np.zeros((ctrl_pts,len(harmonic_test)))
        SPL_BM_unweighted = np.zeros_like(SPL_GD_unweighted)
        SPL_H_unweighted  = np.zeros_like(SPL_GD_unweighted)
        SPL_v_unweighted  = np.zeros_like(SPL_GD_unweighted)
    
    # iterate through sources 
    for source in conditions.noise.sources.keys():
        
        # perform computation of source is propeller or rotor
        if ((source  == 'propeller') or (source  == 'rotor')) and bool(conditions.noise.sources[source]): 
            noise_src = conditions.noise.sources[source]    
            
            # loop for microphone locations 
            for ml in range(num_mic):
                
                # loop for control points  
                for i in range(ctrl_pts):   
                    total_p_pref_r_GD         = []
                    total_p_pref_r_BM         = []
                    total_p_pref_r_H          = []  
                    total_p_pref_GDv_dBA      = []
                    total_p_pref_BMv_dBA      = []
                    total_p_pref_Hv_dBA       = []
                     
                    auc_opts = noise_src.acoustic_outputs 
                    
                    if harmonic_test.any():
                        harmonics    = harmonic_test
                    else:
                        harmonics    = np.arange(1,21) 
                        
                    num_h = len(harmonics)
                     
                    SPL_r_GD        = np.zeros(num_h)
                    SPL_r_BM        = np.zeros_like(SPL_r_GD)
                    SPL_r_H         = np.zeros_like(SPL_r_GD) 
                    p_pref_r_GD     = np.zeros_like(SPL_r_GD)
                    p_pref_r_BM     = np.zeros_like(SPL_r_GD)
                    p_pref_r_H      = np.zeros_like(SPL_r_GD) 
                    SPL_r_GD_dBA    = np.zeros_like(SPL_r_GD)
                    SPL_r_BM_dBA    = np.zeros_like(SPL_r_GD)
                    SPL_r_H_dBA     = np.zeros_like(SPL_r_GD)  
                    p_pref_r_GD_dBA = np.zeros_like(SPL_r_GD)
                    p_pref_r_BM_dBA = np.zeros_like(SPL_r_GD)
                    p_pref_r_H_dBA  = np.zeros_like(SPL_r_GD)
                    
                    for h in range(num_h):
                        m              = harmonics[h]                          # harmonic number 
                        p_ref          = 2e-5                                  # referece atmospheric pressure
                        a              = freestream.speed_of_sound[i][0]       # speed of sound
                        rho            = freestream.density[i][0]              # air density 
                        x              = microphone_location[i,ml,0]           # x relative position from observer
                        y              = microphone_location[i,ml,1]           # y relative position from observer 
                        z              = microphone_location[i,ml,2]           # z relative position from observer
                        Vx             = velocity_vector[i][0]                 # x velocity of propeller  
                        Vy             = velocity_vector[i][1]                 # y velocity of propeller 
                        Vz             = velocity_vector[i][2]                 # z velocity of propeller 
                        thrust_angle   = auc_opts.thrust_angle                 # propeller thrust angle
                        AoA            = angle_of_attack[i][0]                 # vehicle angle of attack                                            
                        N              = auc_opts.number_of_engines            # numner of Rotors
                        B              = auc_opts.number_of_blades             # number of rotor blades
                        omega          = auc_opts.omega[i]                     # angular velocity            rpm = (tip speed*60)/(pi*D)     , 1 rad/s = 9.5492965 rpm    
                        T              = auc_opts.blade_thrust[i]              # propeller/rotor blade thrust     
                        T_distribution = auc_opts.blade_thrust_distribution[i] # propeller/rotor blade thrust distribution  
                        dT_dR          = auc_opts.blade_dT_dR[i]               # differential thrust distribution
                        dT_dr          = auc_opts.blade_dT_dr[i]               # nondimensionalized differential thrust distribution 
                        Q              = auc_opts.blade_torque[i]              # propeller/rotor blade torque    
                        Q_distribution = auc_opts.blade_torque_distribution[i] # propeller/rotor blade torque distribution  
                        dQ_dR          = auc_opts.blade_dT_dR[i]               # differential torque distribution
                        dQ_dr          = auc_opts.blade_dT_dr[i]               # nondimensionalized differential torque distribution
                        R              = auc_opts.radius_distribution          # radial location     
                        c              = auc_opts.chord_distribution           # blade chord    
                        beta           = auc_opts.twist_distribution           # twist distribution  
                        t              = auc_opts.max_thickness_distribution   # twist distribution
                        MCA            = auc_opts.mid_chord_aligment           # Mid Chord Alighment 
                        
                        # A-Weighting for Rotational Noise
                        '''A-weighted sound pressure level can be obtained by applying A(f) to the
                        sound pressure level for each harmonic, then adding the results using the
                        method in Appendix B.
                        '''
                        f  = B*omega*m/(2*np.pi)  
                        #------------------------------------------------------------------------
                        # Rotational SPL by Gutin & Deming
                        #------------------------------------------------------------------------ 
                        n        = len(R)
                        R_tip    = R[-1]                             # Rotor Tip Radius           
                        R_hub    = R[0]                              # Rotor Hub Radius  
                        r        = R/R_tip                           # non dimensional radius distribution
                        V_distri = R*omega                           # blade velocity distribution
                        dR       = R[1] - R[0]
                        dr       = r[1] - r[0]                 
                        A        = np.pi*(R_tip**2)                  # rotor Disc Area 
                        S        = np.sqrt(x**2 + y**2 + z**2)       # distance between rotor and the observer    
                        theta    = np.arccos(x/S)                   
                        alpha    = AoA + thrust_angle
                    
                        # sound pressure for loading noise 
                        p_mL_GD= ((m*B*omega)/(2*np.sqrt(2)*np.pi*a*(S)))*np.trapz((((dT_dR)*np.cos(theta) - ((dQ_dR)*a)/(omega*(R**2)))* 
                                                                            jv(m*B,((m*B*omega*R*np.sin(theta))/a))), x = R, dx = dR  )   
                        # sound pressure for thickness noise 
                        p_mT_GD = ((-rho*((m*B*omega)**2)*B)/(3*np.sqrt(2)*np.pi*(S)))*np.trapz(c*t*jv(m*B,((m*B*omega*R*np.sin(theta))/a)), x = R, dx = dR )  
                        p_mT_GD[np.isinf(p_mT_GD)] = 0
                        p_mT_GD[np.isneginf(p_mT_GD)] = 0                            
                        
                        # unweighted rotational sound pressure level
                        SPL_r_GD[h]        = 10*np.log10(N*((p_mL_GD**2 + p_mT_GD**2 )/p_ref**2))
                        p_pref_r_GD[h]     = 10**(SPL_r_GD[h]/10)  
                        SPL_r_GD_dBA[h]    = A_weighting(SPL_r_GD[h],f)                
                        p_pref_r_GD_dBA[h] = 10**(SPL_r_GD_dBA[h]/10)   
                         
                        #------------------------------------------------------------------------
                        # Rotational SPL  by Barry & Magliozzi
                        #------------------------------------------------------------------------
                        Y            = np.sqrt(y**2 + z**2)                 # observer distance from propeller axis         
                        X            = np.sqrt(x**2 + z**2)                 # distance to observer from propeller plane (z)
                        V            = np.sqrt(Vx**2 + Vy**2 + Vz**2)       # velocity magnitude
                        M            = V/a                                  # Mach number
                        S0           = np.sqrt(x**2 + (1 - M**2)*(Y**2))    # amplitude radius    
                        A_x          = 0.6853*c*t                           # airfoil cross-sectional area
                        # compute phi_t, blade twist angle relative to propeller plane
                        phi_t = np.zeros(n)
                        for idx in range(n):
                            if beta[idx] > 0:
                                phi_t[idx] = np.pi/2 - beta[idx]
                            else:
                                phi_t[idx] = np.pi/2 + abs(beta[idx])               
                                
                        # sound pressure for loading noise 
                        p_mL_BM  = (1/(np.sqrt(2)*np.pi*S0))*np.trapz(( (R/(c*np.cos(phi_t)))*np.sin((m*B*c*np.cos(phi_t))/(2*R))* \
                                                             ((((M + X/S0)*omega)/(a*(1 - M**2)))*(dT_dR) - (1/R**2)*(dQ_dR))* \
                                                             ( (jv(m*B,((m*B*omega*R*Y)/(a*S0)))) + (((1 - M**2)*Y*R)/(2*(S0**2)))*\
                                                              ((jv((m*B-1),((m*B*omega*R*Y)/(a*S0)))) - (jv((m*B+1),((m*B*omega*R*Y)/(a*S0)))))  ) ), x = R, dx = dR )   
                        
                        # sound pressure for thickness noise 
                        p_mT_BM = -((rho*(m**2)*(omega**2)*(B**3) )/(2*np.sqrt(2)*np.pi*((1 - M**2)**2))) *(((S0 + M*X)**2)/(S0**3))* \
                                       np.trapz((A_x *((jv(m*B,((m*B*omega*R*Y)/(a*S0)))) + (((1 - M**2)*Y*R)/(2*(S0**2)))*\
                                       ((jv((m*B-1),((m*B*omega*R*Y)/(a*S0)))) - (jv((m*B+1),((m*B*omega*R*Y)/(a*S0))))) )), x = R, dx = dR )
                        p_mT_BM[np.isinf(p_mT_BM)] = 0
                        p_mT_BM[np.isneginf(p_mT_BM)] = 0
                        
                        # unweighted rotational sound pressure level          
                        SPL_r_BM[h]        = 10*np.log10(N*((p_mL_BM**2 + p_mT_BM**2 )/p_ref**2))
                        p_pref_r_BM[h]     = 10**(SPL_r_BM[h]/10)  
                        SPL_r_BM_dBA[h]    = A_weighting(SPL_r_BM[h],f)
                        p_pref_r_BM_dBA[h] = 10**(SPL_r_BM_dBA[h]/10)   
                       
                        
                        #------------------------------------------------------------------------
                        # Rotational SPL by Hanson
                        #------------------------------------------------------------------------
                        D             = 2*R_tip                                                                                 # propeller diameter
                        V_tip         = R_tip*omega                                                                             # blade_tip_speed 
                        M_t           = V_tip/a                                                                                 # tip Mach number 
                        M_s           = np.sqrt(M**2 + (r**2)*(M_t**2))                                                         # section relative Mach number 
                        r_t           = R_tip                                                                                   # propeller tip radius
                        phi           = np.arctan(z/y)                                                                          # tangential angle  
                        theta_r       = np.arccos(np.cos(theta)*np.sqrt(1 - (M**2)*(np.sin(theta))**2) + M*(np.sin(theta))**2 ) # theta angle in the retarded reference frame
                        theta_r_prime = np.arccos(np.cos(theta_r)*np.cos(alpha) + np.sin(theta_r)*np.sin(phi)*np.sin(alpha) )   #
                        phi_prime     = np.arccos((np.sin(theta_r)/np.sin(theta_r_prime))*np.cos(phi))                          # phi angle relative to propeller shaft axis                                                   
                        phi_s         = ((2*m*B*M_t)/(M_s*(1 - M*np.cos(theta_r))))*(MCA/D)                                     # phase lag due to sweep
                        S_r           = Y/(np.sin(theta_r))                                                                     # distance in retarded reference frame 
                        k_x           = ((2*m*B*c*M_t)/(M_s*(1 - M*np.cos(theta_r))))                                           # wave number 
                        psi_L         = np.zeros(n)
                        psi_V         = np.zeros(n)
                        for idx in range(n):
                            if k_x[idx] == 0:                        
                                psi_V[idx] = 2/3                                                                                # normalized thickness souce transforms
                                psi_L[idx] = 1                                                                                  # normalized loading souce transforms
                            else:  
                                psi_V[idx] = (8/(k_x[idx]**2))*((2/k_x[idx])*np.sin(0.5*k_x[idx]) - np.cos(0.5*k_x[idx]))       # normalized thickness souce transforms           
                                psi_L[idx] = (2/k_x[idx])*np.sin(0.5*k_x[idx])                                                  # normalized loading souce transforms
                        # sound pressure for loading noise 
                        p_mL_H = ((  1j*m*B*M_t*np.sin(theta_r)*np.exp(1j*m*B*((omega*S_r/a)+(phi_prime - np.pi/2))) ) / (2*np.sqrt(2)*np.pi*Y*r_t*(1 - M*np.cos(theta_r)))  ) \
                                  *np.trapz(( (   (np.cos(theta_r_prime)/(1 - M*np.cos(theta_r)))*(dT_dr) - (1/((r**2)*M_t*r_t))*(dQ_dr)  ) * np.exp(1j*phi_s) *\
                                       (jv(m*B,((m*B*r*M_t*np.sin(theta_r_prime))/(1 - M*np.cos(theta_r))))) * psi_L  ),x = r,dx = dr)
                        p_mL_H[np.isinf(p_mL_H)] = 0
                        p_mL_H = abs(p_mL_H)
                        
                        # sound pressure for thickness noise 
                        p_mT_H = (-(rho*(a**2)*B*np.sin(theta_r)*np.exp(1j*m*B*((omega*S_r/a)+(phi_prime - np.pi/2))))/(4*np.sqrt(2)*np.pi*(Y/D)*(1 - M*np.cos(theta_r)))) \
                                *np.trapz(((M_s**2)*(t/c)*np.exp(1j*phi_s)*(jv(m*B,((m*B*r*M_t*np.sin(theta_r_prime))/(1 - M*np.cos(theta_r)))))*(k_x**2)*psi_V ),x = r,dx = dr)
                        p_mT_H[np.isinf(p_mT_H)] = 0  
                        p_mT_H  = abs(p_mT_H)
                        
                        # unweighted rotational sound pressure level
                        SPL_r_H[h]        = 10*np.log10(N*((p_mL_H**2 + p_mT_H**2 )/p_ref**2))  
                        p_pref_r_H[h]     = 10**(SPL_r_H[h]/10)  
                        SPL_r_H_dBA[h]    = A_weighting(SPL_r_H[h],f)
                        p_pref_r_H_dBA[h] = 10**(SPL_r_H_dBA[h]/10)   
                        
                    # -----------------------------------------------------------------------
                    # Vortex Noise (developed by Schlegel et. al in Helicopter Rotor Noise) This is computed in feet 
                    # ----------------------------------------------------------------------- 
                    V_07      = V_tip*0.70/(Units.feet)                     # blade velocity at r/R_tip = 0.7 
                    St        = 0.28                                        # Strouhal number             
                    t_avg     = np.mean(t)/(Units.feet)                                  # thickness
                    c_avg     = np.mean(c)/(Units.feet)                                  # average chord  
                    beta_07   = beta[round(n*0.70)]                                      # blade angle of attack at r/R = 0.7
                    h_val     = t_avg*np.cos(beta_07) + c_avg*np.sin(beta_07)            # projected blade thickness                   
                    f_peak    = (V_07*St)/h_val                                          # V - blade velocity at a radial location of 0.7              
                    A_blade   = (np.trapz(c, dx = dR))/(Units.feet**2) # area of blade
                    CL_07     = 2*np.pi*beta_07
                    S_feet    = S/(Units.feet)
                    SPL_300ft = 10*np.log10(((6.1e-27)*A_blade*V_07**6)/(10**-16)) + 20*np.log(CL_07/0.4) # CORRECT 
                    SPL_v     = SPL_300ft - 20*np.log10(S_feet/300)   # CORRECT 
                    
                    # estimation of A-Weighting for Vortex Noise  
                    f_spectrum  = [0.5*f_peak, 1*f_peak , 2*f_peak , 4*f_peak , 8*f_peak , 16*f_peak]  # spectrum
                    fr          = f_spectrum/f_peak                                                    # frequency ratio  
                    SPL_weight  = [7.92 , 4.17 , 8.33 , 8.75 ,12.92 , 13.33]                           # SPL weight
                    SPL_v       = np.ones_like(SPL_weight)*SPL_v - SPL_weight                          # SPL correction
                    
                    dim         = len(f_spectrum)
                    C           = np.zeros(dim) 
                    p_pref_v_dBA= np.zeros(dim-1)
                    SPL_v_dbAi = np.zeros(dim)
                    
                    for j in range(dim):
                        SPL_v_dbAi[j] = A_weighting(SPL_v[j],f_spectrum[j])
                    
                    for j in range(dim-1):
                        C[j]        = (SPL_v_dbAi[j+1] - SPL_v_dbAi[j])/(np.log10(fr[j+1]) - np.log10(fr[j])) 
                        C[j+1]      = SPL_v_dbAi[j+1] - C[j]*np.log10(fr[j+1])   
                        p_pref_v_dBA[j] = (10**(0.1*C[j+1]))* (  ((fr[j+1]**(0.1*C[j] + 1 ))/(0.1*C[j] + 1 )) - ((fr[j]**(0.1*C[j] + 1 ))/(0.1*C[j] + 1 )) )
                        
                        
                    # collecting unweighted pressure ratios  
                    total_p_pref_r_GD.append(p_pref_r_GD)  
                    total_p_pref_r_BM.append(p_pref_r_BM)  
                    total_p_pref_r_H.append(p_pref_r_H)    
                    
                    # collecting weighted pressure ratios with vortex noise included 
                    total_p_pref_GDv_dBA.append(np.concatenate([p_pref_r_GD_dBA,p_pref_v_dBA])) # Gutin & Deming rotational noise with Schlegel vortex noise
                    total_p_pref_BMv_dBA.append(np.concatenate([p_pref_r_BM_dBA,p_pref_v_dBA])) # Barry & Magliozzi rotational noise with Schlegel vortex noise
                    total_p_pref_Hv_dBA.append(np.concatenate([p_pref_r_H_dBA,p_pref_v_dBA]))   # Hanson rotational noise with Schlegel vortex noise        
                    
                    
                    SPL_v_dBA[i,ml] = pressure_ratio_to_SPL_arithmetic(p_pref_v_dBA)
                    
                    if harmonic_test.any(): 
                        SPL_GD_unweighted[i,ml] = SPL_r_GD
                        SPL_BM_unweighted[i,ml] = SPL_r_BM 
                        SPL_H_unweighted[i,ml]  = SPL_r_H 
                        
                    else:
                        # Rotational SPL (Unweighted)    
                        SPL_GD_unweighted[i,ml]      = pressure_ratio_to_SPL_arithmetic(total_p_pref_r_GD)
                        SPL_BM_unweighted[i,ml]      = pressure_ratio_to_SPL_arithmetic(total_p_pref_r_BM)
                        SPL_H_unweighted[i,ml]       = pressure_ratio_to_SPL_arithmetic(total_p_pref_r_H)  
                        #SPL_v_unweighted[i,ml]       = SPL_v
                    
                    # A- Weighted Rotational and Vortex SPL 
                    SPL_GDv_dBA[i,ml]      = pressure_ratio_to_SPL_arithmetic(total_p_pref_GDv_dBA)
                    SPL_BMv_dBA[i,ml]      = pressure_ratio_to_SPL_arithmetic(total_p_pref_BMv_dBA)
                    SPL_Hv_dBA[i,ml]       = pressure_ratio_to_SPL_arithmetic(total_p_pref_Hv_dBA)
            
            # append SPL of source      
            noise_src.SPL_GD_unweighted = SPL_GD_unweighted
            noise_src.SPL_BM_unweighted = SPL_BM_unweighted
            noise_src.SPL_H_unweighted  = SPL_H_unweighted 
            noise_src.SPL_v_unweighted  = SPL_v_unweighted 
            noise_src.SPL_GDv_dBA       = SPL_GDv_dBA      
            noise_src.SPL_BMv_dBA       = SPL_BMv_dBA      
            noise_src.SPL_Hv_dBA        = SPL_Hv_dBA       
            noise_src.SPL_v_dBA         = SPL_v_dBA     
    
    return   
 
