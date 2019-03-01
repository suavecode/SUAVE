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

def noise_propeller_low_fidelty(noise_data,ctrl_pts):
    """Inputs:
           noise_data	 - SUAVE type vehicle

       Outputs:
           OASPL                         - Overall Sound Pressure Level, [dB]
           PNL                           - Perceived Noise Level, [dB]
           PNL_dBA                       - Perceived Noise Level A-weighted level, [dBA]

       Assumptions:
           Empirical based procedure."""   
    
    
    SPL_DG_unweighted2 = np.zeros((ctrl_pts,1)) 
    SPL_DG_unweighted = np.zeros((ctrl_pts,1)) 
    SPL_BM_unweighted = np.zeros((ctrl_pts,1))
    SPL_H_unweighted  = np.zeros((ctrl_pts,1)) 
    SPL_GDv_dBA2       = np.zeros((ctrl_pts,1))
    SPL_GDv_dBA       = np.zeros((ctrl_pts,1))
    SPL_BMv_dBA       = np.zeros((ctrl_pts,1))
    SPL_Hv_dBA        = np.zeros((ctrl_pts,1))      

    for i in range(ctrl_pts): 
 
        total_p_pref_r_GD = []
        total_p_pref_r_GD2 = []
        total_p_pref_r_BM = []
        total_p_pref_r_H  = []
        
        total_p_pref_GDv_dBA = []
        total_p_pref_GDv_dBA2 = []
        total_p_pref_BMv_dBA = []
        total_p_pref_Hv_dBA = []
        
        for prop  in noise_data.values(): 
            harmonics    = range(1,2)  
            num_h        = len(harmonics)
            
            SPL_r_GD2     = np.zeros((num_h,1))
            SPL_r_GD     = np.zeros((num_h,1))
            SPL_r_BM     = np.zeros((num_h,1))
            SPL_r_H      = np.zeros((num_h,1))
            
            p_pref_r_GD2  = np.zeros((num_h,1))
            p_pref_r_GD  = np.zeros((num_h,1))
            p_pref_r_BM  = np.zeros((num_h,1))
            p_pref_r_H   = np.zeros((num_h,1))
            
            SPL_r_GD_dBA2 = np.zeros((num_h,1))
            SPL_r_GD_dBA = np.zeros((num_h,1))
            SPL_r_BM_dBA = np.zeros((num_h,1))
            SPL_r_H_dBA  = np.zeros((num_h,1))
            
            p_pref_r_GD_dBA2 = np.zeros((num_h,1))
            p_pref_r_GD_dBA = np.zeros((num_h,1))
            p_pref_r_BM_dBA = np.zeros((num_h,1))
            p_pref_r_H_dBA  = np.zeros((num_h,1))

            for h in range(num_h):
                m              = harmonics[h]                   # harmonic number 
                p_ref          = 2e-5                           # referece atmospheric pressure
                a              = prop.speed_of_sound[i][0]         # speed of sound
                rho            = prop.density[i][0]                # air density 
                x              = prop.position[i][0]            # x relative position from observer
                y              = prop.position[i][1]            # y relative position from observer 
                z              = prop.position[i][2]            # z relative position from observer
                Vx             = prop.velocity[i][0]            # x velocity of propeller  
                Vy             = prop.velocity[i][1]            # y velocity of propeller 
                Vz             = prop.velocity[i][2]            # z velocity of propeller 
                thrust_angle   = prop.thrust_angle              # propeller thrust angle
                AoA            = prop.AoA[i][0]                    # vehicle angle of attack                                            
                N              = prop.n_rotors                  # numner of Rotors
                B              = prop.n_blades                  # number of rotor blades
                omega          = prop.omega[i]                  # angular velocity            rpm = (tip speed*60)/(pi*D)     , 1 rad/s = 9.5492965 rpm     
                T              = prop.thrust[i]                 # rotor thrust                T (supposed) 14011 (total)  1751 (per rotor)  
                Q              = prop.torque[i]                 # rotor torque  
                T_distribution = prop.thrust_distribution[i]    # rotor thrust distribution  
                Q_distribution = prop.torque_distribution[i]    # rotor torque distribution 
                R           = prop.radius_distribution       # radial location     
                c           = prop.chord_distribution        # blade chord    
                phi_t         = prop.twist_distribution        # twist distribution  
                t           = prop.max_thickness_distribution # twist distribution
                MCA         = prop.mid_chord_aligment        # Mid Chord Alighment 
                
                # A-Weighting for Rotational Noise
                '''A-weighted sound pressure level can be obtained by applying A(f) to the
                sound pressure level for each harmonic, then adding the results using the
                method in Appendix B.'''
                f  = B*omega*m/(2*np.pi)  
                
                #------------------------------------------------------------------------
                # Rotational SPL by Gutin & Deming
                #------------------------------------------------------------------------  
                R_tip          = R[-1]                      # Rotor Tip Radius           
                R_hub          = R[0]                       # Rotor Hub Radius  
                r          = R/R_tip                    # non dimensional radius distribution
                V_distribution = R*omega                    # blade velocity distribution
                 
                n   = len(R) 
                A   = np.pi*(R_tip**2)               # rotor Disc Area  
                S   = np.sqrt(x**2 + y**2 + z**2)    # distance between rotor and the observer 
                
                # compute theta, the azimuthal angle location 
                if Vx == 0:
                    theta = np.pi - np.arcsin(y/S)  #   np.arccos(x/S) 
                    
                    alpha = 0. 
                else:
                    v1    = [Vx,Vy,Vz]  # velocity_vector
                    v2    = [x,y,z]     # observer_vector   
                    theta = np.arccos(np.dot(v1,v2)/ (np.linalg.norm(v1) * np.linalg.norm(v2))) 
                    #theta2 = np.arccos(x/S) 
                    alpha = AoA
                print('theta')
                print(theta/Units.degrees)
                #print(theta2/Units.degrees) 
                
                
                V              = np.sqrt(Vx**2 + Vy**2 + Vz**2) # velocity magnitude               
                M              = V/a                            # Mach number
                Re             = 0.7*R_tip
                V_07           = V_distribution[13]             # blade velocity at r/R_tip = 0.7        
                V_tip          = R_tip*omega                    # blade_tip_speed         

                # sound pressure for loading noise 
                p_mL_GD = ((m*B*omega)/(2*np.sqrt(2)*np.pi*a*(S)))*np.sum((T_distribution*np.cos(theta) - (Q_distribution*a)/(omega*(R**2)))* 
                                                                       jv(m*B,((m*B*omega*R*np.sin(theta))/a)))                
                #p_mL_GD[np.isinf(p_mL_GD)] = 0
                
                # sound pressure for thickness noise 
                p_mT_GD = -((rho*((m*B*omega)**2)*B)/(3*np.sqrt(2)*np.pi*(S)))*np.sum(c*t*R*jv(m*B,((m*B*omega*R*np.sin(theta))/a)))
                #p_mT_GD[np.isinf(p_mT_GD)] = 0
                #p_mT_GD[np.isneginf(p_mT_GD)] = 0
                
                # unweighted rotational sound pressure level
                SPL_r_GD[h]        = 10*np.log10(N*((p_mL_GD**2 + p_mT_GD**2 )/p_ref**2))
                p_pref_r_GD[h]     = 10**(SPL_r_GD[h]/10)  
                SPL_r_GD_dBA[h]    = A_weighting(SPL_r_GD[h],f)                
                p_pref_r_GD_dBA[h] = 10**(SPL_r_GD_dBA[h]/10)   
                
                #------------------------------------------------------------------------
                # Rotational SPL by Gutin & Deming SIMPLIFIED
                #------------------------------------------------------------------------  
                
                # sound pressure for loading noise 
                p_mL_GD2 = ((m*B*omega)/(2*np.sqrt(2)*np.pi*a*(S)))*(T*np.cos(theta) - (Q*a)/(omega*(Re**2)))* jv(m*B,((m*B*omega*Re*np.sin(theta))/a))                
                p_mL_GD2[np.isinf(p_mL_GD2)] = 0
                
                # sound pressure for thickness noise 
                p_mT_GD2 = -((rho*((m*B*omega)**2)*B)/(3*np.sqrt(2)* np.pi*(S))) * np.mean(c) * Re* np.mean(t) *jv(m*B,((m*B*omega*Re*np.sin(theta))/a))
                p_mT_GD2[np.isinf(p_mT_GD2)] = 0
                p_mT_GD2[np.isneginf(p_mT_GD2)] = 0
                
                # unweighted rotational sound pressure level
                SPL_r_GD2[h]        = 10*np.log10(N*((p_mL_GD2**2 + p_mT_GD2**2 )/p_ref**2))
                p_pref_r_GD2[h]     = 10**(SPL_r_GD2[h]/10)  
                SPL_r_GD_dBA2[h]    = A_weighting(SPL_r_GD2[h],f)                
                p_pref_r_GD_dBA2[h] = 10**(SPL_r_GD_dBA2[h]/10)  
                
                #------------------------------------------------------------------------
                # Rotational SPL  by Barry & Magliozzi
                #------------------------------------------------------------------------
                Y            = np.sqrt(y**2 + z**2)                 # observer distance from propeller axis         
                X            = np.sqrt(x**2 + z**2)                 # distance to observer from propeller plane (z)  
                S0           = np.sqrt(z**2 + (1 - M**2)*(Y**2))    # amplitude radius                                
                J_mB         = jv(m*B,((m*B*omega*R*Y)/a))          # Bessel function of order mB and argument x 
                J_mB_plus_1  = jv((m*B+1),((m*B*omega*R*Y)/(a*S0))) # Bessel function of order mB and argument x 
                J_mB_minus_1 = jv((m*B-1),((m*B*omega*R*Y)/(a*S0))) # Bessel function of order mB and argument x   
                A_x          = 0.6853*c*t                       # airfoil cross-sectional area
                
                # sound pressure for loading noise 
                p_mL_BM = (1/(np.sqrt(2)*np.pi*S0))*np.sum( (R/(c*np.cos(phi_t)))*np.sin((m*B*c*np.cos(phi_t))/(2*R))* \
                                                         ((((M + X/S0)*omega)/(a*(1 - M**2)))*T_distribution - (1/R**2)*Q_distribution)* \
                                                         (J_mB + (((1 - M**2)*Y*R)/(2*(S0**2)))*(J_mB_minus_1 - J_mB_plus_1)))               
                
                #p_mL_BM[np.isinf(p_mL_BM)] = 0
                
                # sound pressure for thickness noise 
                p_mT_BM = -((rho*(m**2)*(omega**2)*(B**3) )/(2*np.sqrt(2)*np.pi*((1 - M**2)**2)))*(((S0 + M*X)**2)/(S0**3))* np.sum(A_x *(J_mB + (((1 - M**2)*Y*R)/(2*(S0**2)))*(J_mB_minus_1 - J_mB_plus_1))) 
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
                D             = 2*R_tip                             # propeller diameter
                M_t           = V_tip/a                             # tip Mach number 
                M_s           = np.sqrt(M**2 + (r**2)*(M_t**2))     # section relative Mach number 
                r_t           = R_tip                               # propeller tip radius
                phi           = np.arctan(z/y)            # tangential angle     np.arctan(z/y) 
                
                print('phi')
                print(phi/Units.degrees)
                
                theta_r       = np.arccos(np.cos(theta)*np.sqrt(1 - (M**2)*(np.sin(theta))**2) + M*(np.sin(theta))**2 )      # theta angle in the retarded reference frame
                theta_r_prime = np.arccos(np.cos(theta_r)*np.cos(alpha) + np.sin(theta_r)*np.sin(phi)*np.sin(alpha) )  #
                phi_prime     = np.arccos((np.sin(theta_r)/np.sin(theta_r_prime))*np.cos(phi))                         # phi angle relative to propeller shaft axis                                                   
                phi_s         = ((2*m*B*M_t)/(M_s*(1 - M*np.cos(theta_r))))*(MCA/D)                                   # phase lag due to sweep
                J_mB          = jv(m*B,((m*B*r*omega*M_t*np.sin(theta_r_prime))/(1 - M*np.cos(theta_r))))              # Bessel function of order mB and argument x
                S_r           = Y/(np.sin(theta_r))                                                                    # distance in retarded reference frame 
                k_x           = ((2*m*B*c*M_t)/(M_s*(1 - M*np.cos(theta_r))))                                          # wave number 
                psi_L         = np.zeros((n,1))
                psi_V         = np.zeros((n,1))
                for idx in range(n):
                    if k_x[idx] == 0:
                        psi_L[idx] = 2/3                                                                              # normalized loading souce transforms
                        psi_V[idx] = (8/k_x[idx]**2)*((2/k_x[idx])*np.sin(0.5*k_x[idx])- np.cos(0.5*k_x[idx]))        # normalized thickness souce transforms
                    else:                                                                                             
                        psi_L[idx] = 1                                                                                # normalized loading souce transforms
                        psi_V[idx] = (2/k_x[idx])*np.sin(0.5*k_x[idx])                                                # normalized thickness souce transforms           
                
                # sound pressure for loading noise 
                p_mL_H = ((1j*m*B*M_t*np.sin(theta_r)*np.exp(1j*m*B*((omega*S_r/a)+(phi_prime - np.pi/2))))/(2*np.sqrt(2)*np.pi*Y*r_t*(1 - M*np.cos(theta_r)))) \
                    * np.sum(((np.cos(theta_r_prime)/(1 - M*np.cos(theta_r)))*T_distribution - (1/((r**2)*M_t*r_t))*Q_distribution)*np.exp(1j*phi_s)*J_mB*psi_L)
                                    
                p_mL_H[np.isinf(p_mL_H)] = 0
                p_mL_H = p_mL_H.real
                
                # sound pressure for thickness noise 
                p_mT_H = (-(rho*(a**2)*B*np.sin(theta_r)*np.exp(1j*m*B*((omega*S_r/a)+(phi_prime - np.pi/2))))/(4*np.sqrt(2)*np.pi*(Y/D)*(1 - M*np.cos(theta_r)))) \
                    * np.sum((M_s**2)*(t/c)*np.exp(1j*phi_s)*J_mB*(k_x**2)*psi_V) 
                p_mT_H[np.isinf(p_mT_H)] = 0  
                p_mT_H = p_mT_H.real
                
                # unweighted rotational sound pressure level
                SPL_r_H[h]        = 10*np.log10(N*((p_mL_H**2 + p_mT_H**2 )/p_ref**2))  
                p_pref_r_H[h]     = 10**(SPL_r_H[h]/10)  
                SPL_r_H_dBA[h]    = A_weighting(SPL_r_H[h],f)
                p_pref_r_H_dBA[h] = 10**(SPL_r_H_dBA[h]/10)   
                               
                
            # -----------------------------------------------------------------------
            # Vortex Noise (developed by Schlegel et. al)
            # ----------------------------------------------------------------------- 
            St       = 0.28                                        # Strouhal number             
            t_avg    = np.mean(t)                                  # thickness
            c_avg    = np.mean(c)                                  # average chord  
            A_blade  = np.trapz(c, dx = (R[1] - R[0])) # area of blade
            s        = (B*A_blade)/A                               # rotor solidity
            K        = 6.1e-11    
            K_2      = np.sqrt((0.7**2)*(15**2)*K*S**2)               
            
            # unweighted overall vortex sound pressure level
            SPL_v    = 20*np.log10((K_2)*(V_tip/(rho*S))*np.sqrt(((N*T)/s)*(T/A)))
            SPL_v[np.isnan(SPL_v)] = 0
            
            # estimation of A-Weighting for Vortex Noise  
            Cl_bar      = (6*T)/(rho*s*V_tip*A)                                                # mean lift coefficient
            blade_alpha = Cl_bar/(2*np.pi)                                                     # blade angle of attack at r/R = 0.7
            h           = t_avg*np.cos(blade_alpha) + c_avg*np.sin(blade_alpha)                # projected blade thickness                   
            f_peak      = (V_07*St)/h                                                          # V - blade velocity at a radial location of 0.7                
            f_spectrum  = [0.5*f_peak, 1*f_peak , 2*f_peak , 4*f_peak , 8*f_peak , 16*f_peak]  # spectrum
            SPL_weight  = [7.92 , 4.17 , 8.33 , 8.75 ,12.92 , 13.33]                           # SPL weight
            SPL_v       = np.ones_like(SPL_weight)*SPL_v - SPL_weight                          # SPL correction
            SPL_v_dBA   = np.zeros((len(SPL_v),1))
            for idx in range(len(SPL_v)):
                SPL_v_dBA[idx] = A_weighting(SPL_v[idx],f)
            
            dim         = len(f_spectrum)
            C           = np.zeros(dim) 
            fr          = f_spectrum/f_peak                                                             # frequency ratio  
            p_pref_v_dBA= np.zeros((len(SPL_v),1))
            
            # A-Weighting for Vortex Noise  
            for j in range(dim-1):
                C[j]        = (SPL_v[j+1] - SPL_v[j])/(np.log10(fr[j+1]) - np.log10(fr[j])) 
                C[j+1]      = SPL_v[j+1]- C[j]*np.log10(fr[j+1])   
                p_pref_v_dBA[j] = (10** (0.1*C[j+1]))* (((fr[j+1]**(0.1*C[j] + 1 ))/(0.1*C[j] + 1 )) - ((fr[j]**(0.1*C[j] + 1 ))/(0.1*C[j] + 1 )) )
                    
            # collecting unweighted pressure ratios 
            total_p_pref_r_GD2.append(p_pref_r_GD2) 
            total_p_pref_r_GD.append(p_pref_r_GD)  
            total_p_pref_r_BM.append(p_pref_r_BM)  
            total_p_pref_r_H.append(p_pref_r_H)    
            
            # collecting weighted pressure ratios with vortex noise included
            total_p_pref_GDv_dBA2.append(np.concatenate([p_pref_r_GD_dBA2,p_pref_v_dBA])) # Gutin & Deming rotational noise with Schlegel vortex noise
            total_p_pref_GDv_dBA.append(np.concatenate([p_pref_r_GD_dBA,p_pref_v_dBA])) # Gutin & Deming rotational noise with Schlegel vortex noise
            total_p_pref_BMv_dBA.append(np.concatenate([p_pref_r_BM_dBA,p_pref_v_dBA])) # Barry & Magliozzi rotational noise with Schlegel vortex noise
            total_p_pref_Hv_dBA.append(np.concatenate([p_pref_r_H_dBA,p_pref_v_dBA])) # Hanson rotational noise with Schlegel vortex noise        
        
        # Rotational SPL (Unweighted)   
        SPL_DG_unweighted2[i,0] = decibel_arithmetic(total_p_pref_r_GD2)
        SPL_DG_unweighted[i,0] = decibel_arithmetic(total_p_pref_r_GD)
        SPL_BM_unweighted[i,0] = decibel_arithmetic(total_p_pref_r_BM)
        SPL_H_unweighted[i,0]  = decibel_arithmetic(total_p_pref_r_H)            
        
        # A- Weighted Rotational and Vortex SPL
        SPL_GDv_dBA2[i,0] = decibel_arithmetic( total_p_pref_GDv_dBA2)
        SPL_GDv_dBA[i,0] = decibel_arithmetic( total_p_pref_GDv_dBA)
        SPL_BMv_dBA[i,0] = decibel_arithmetic(total_p_pref_BMv_dBA)
        SPL_Hv_dBA[i,0]  = decibel_arithmetic(total_p_pref_Hv_dBA)
    
    return   SPL_DG_unweighted2 , SPL_DG_unweighted , SPL_BM_unweighted , SPL_H_unweighted , SPL_GDv_dBA2 , SPL_GDv_dBA  , SPL_BMv_dBA , SPL_Hv_dBA
# -----------------------------------------------------------------------
# Decibel Arithmetic
# -----------------------------------------------------------------------
def decibel_arithmetic(p_pref_total):
    SPL_total = 10*np.log10(np.sum(p_pref_total))
    return SPL_total

# -----------------------------------------------------------------------
# Rotational Noise A-Weight
# -----------------------------------------------------------------------
def  A_weighting(SPL,f):
    Ra_f       = ((12200**2)*(f**4))/ (((f**2)+(20.6**2)) * ((f**2)+(12200**2)) * (((f**2) + 107.7**2)**0.5)* (((f**2)+ 737.9**2)**0.5)) 
    A_f        =  2.0  + 20*np.log10(Ra_f) 
    SPL_dBA = SPL + A_f 
    return SPL_dBA












#def noise_propeller_low_fidelty(noise_data,ctrl_pts):
    #"""Inputs:
           #noise_data	 - SUAVE type vehicle

       #Outputs:
           #OASPL                         - Overall Sound Pressure Level, [dB]
           #PNL                           - Perceived Noise Level, [dB]
           #PNL_dBA                       - Perceived Noise Level A-weighted level, [dBA]

       #Assumptions:
           #Empirical based procedure."""   
    
    
    #SPL_DG_unweighted2 = np.zeros((ctrl_pts,1)) 
    #SPL_DG_unweighted = np.zeros((ctrl_pts,1)) 
    #SPL_BM_unweighted = np.zeros((ctrl_pts,1))
    #SPL_H_unweighted  = np.zeros((ctrl_pts,1)) 
    #SPL_GDv_dBA2       = np.zeros((ctrl_pts,1))
    #SPL_GDv_dBA       = np.zeros((ctrl_pts,1))
    #SPL_BMv_dBA       = np.zeros((ctrl_pts,1))
    #SPL_Hv_dBA        = np.zeros((ctrl_pts,1))      

    #for i in range(ctrl_pts): 
 
        #total_p_pref_r_GD = []
        #total_p_pref_r_GD2 = []
        #total_p_pref_r_BM = []
        #total_p_pref_r_H  = []
        
        #total_p_pref_GDv_dBA = []
        #total_p_pref_GDv_dBA2 = []
        #total_p_pref_BMv_dBA = []
        #total_p_pref_Hv_dBA = []
        
        #for prop  in noise_data.values(): 
            #harmonics    = range(1,2)  
            #num_h        = len(harmonics)
            
            #SPL_r_GD2     = np.zeros((num_h,1))
            #SPL_r_GD     = np.zeros((num_h,1))
            #SPL_r_BM     = np.zeros((num_h,1))
            #SPL_r_H      = np.zeros((num_h,1))
            
            #p_pref_r_GD2  = np.zeros((num_h,1))
            #p_pref_r_GD  = np.zeros((num_h,1))
            #p_pref_r_BM  = np.zeros((num_h,1))
            #p_pref_r_H   = np.zeros((num_h,1))
            
            #SPL_r_GD_dBA2 = np.zeros((num_h,1))
            #SPL_r_GD_dBA = np.zeros((num_h,1))
            #SPL_r_BM_dBA = np.zeros((num_h,1))
            #SPL_r_H_dBA  = np.zeros((num_h,1))
            
            #p_pref_r_GD_dBA2 = np.zeros((num_h,1))
            #p_pref_r_GD_dBA = np.zeros((num_h,1))
            #p_pref_r_BM_dBA = np.zeros((num_h,1))
            #p_pref_r_H_dBA  = np.zeros((num_h,1))

            #for h in range(num_h):
                #m              = harmonics[h]                   # harmonic number 
                #p_ref          = 2e-5                           # referece atmospheric pressure
                #a              = prop.speed_of_sound[i][0]         # speed of sound
                #rho            = prop.density[i][0]                # air density 
                #x              = prop.position[i][0]            # x relative position from observer
                #y              = prop.position[i][1]            # y relative position from observer 
                #z              = prop.position[i][2]            # z relative position from observer
                #Vx             = prop.velocity[i][0]            # x velocity of propeller  
                #Vy             = prop.velocity[i][1]            # y velocity of propeller 
                #Vz             = prop.velocity[i][2]            # z velocity of propeller 
                #thrust_angle   = prop.thrust_angle              # propeller thrust angle
                #AoA            = prop.AoA[i][0]                    # vehicle angle of attack                                            
                #N              = prop.n_rotors                  # numner of Rotors
                #B              = prop.n_blades                  # number of rotor blades
                #omega          = prop.omega[i]                  # angular velocity            rpm = (tip speed*60)/(pi*D)     , 1 rad/s = 9.5492965 rpm     
                #T              = prop.thrust[i]                 # rotor thrust                T (supposed) 14011 (total)  1751 (per rotor)  
                #Q              = prop.torque[i]                 # rotor torque  
                #T_distribution = prop.thrust_distribution[i]    # rotor thrust distribution  
                #Q_distribution = prop.torque_distribution[i]    # rotor torque distribution 
                #R_old          = prop.radius_distribution       # radial location     
                #c_old          = prop.chord_distribution        # blade chord    
                #phi_t_old      = prop.twist_distribution        # twist distribution  
                #t_old          = prop.max_thickness_distribution # twist distribution
                #MCA_old        = prop.mid_chord_aligment        # Mid Chord Alighment 
                
                ## A-Weighting for Rotational Noise
                #'''A-weighted sound pressure level can be obtained by applying A(f) to the
                #sound pressure level for each harmonic, then adding the results using the
                #method in Appendix B.'''
                #f  = B*omega*m/(2*np.pi)  
                
                ##------------------------------------------------------------------------
                ## Rotational SPL by Gutin & Deming
                ##------------------------------------------------------------------------  
                #R_tip          = R_old[-1]                      # Rotor Tip Radius           
                #R_hub          = R_old[0]                       # Rotor Hub Radius  
                #r_old          = R_old/R_tip                    # non dimensional radius distribution
                #V_distribution = R_old*omega                    # blade velocity distribution
                #dT_dR = np.diff(T_distribution)/np.diff(R_old)  # dT_dR 
                #dT_dr = np.diff(T_distribution)/np.diff(r_old)  # dT_dr
                #dQ_dR = np.diff(Q_distribution)/np.diff(R_old)  # dQ_dR            
                #dQ_dr = np.diff(Q_distribution)/np.diff(r_old)  # dQ_dr
                
                ## we have to reduce dimension of R, r and c since dimension of differentials are 1 less 
                #n   = len(R_old) - 1
                #R   = np.linspace(R_hub,R_tip,n)    # Radius distribution
                #r   = np.interp(R,R_old,r_old)       # non-dimensional radius distribution
                #c   = np.interp(R,R_old,c_old)       # chord distribution
                #T_distribution   = np.interp(R,R_old,T_distribution) 
                #Q_distribution   = np.interp(R,R_old,Q_distribution)        
                #t   = np.interp(R,R_old,t_old)       # max thickness distribution
                #MCA = np.interp(R,R_old,MCA_old)     # MCA distribution
                #A   = np.pi*(R_tip**2)               # rotor Disc Area   
                         
                #S   = np.sqrt(x**2 + y**2 + z**2)    # distance between rotor and the observer 
                
                ## compute theta, the azimuthal angle location 
                #if Vx == 0:
                    #theta = np.pi - np.arcsin(y/S)  #   np.arccos(x/S) 
                    
                    #alpha = 0. 
                #else:
                    #v1    = [Vx,Vy,Vz]  # velocity_vector
                    #v2    = [x,y,z]     # observer_vector   
                    #theta = np.arccos(np.dot(v1,v2)/ (np.linalg.norm(v1) * np.linalg.norm(v2))) 
                    ##theta2 = np.arccos(x/S) 
                    #alpha = AoA
                #print('theta')
                #print(theta/Units.degrees)
                ##print(theta2/Units.degrees) 
                
                
                #V              = np.sqrt(Vx**2 + Vy**2 + Vz**2) # velocity magnitude               
                #M              = V/a                            # Mach number
                #Re             = 0.7*R_tip
                #V_07           = V_distribution[13]             # blade velocity at r/R_tip = 0.7        
                #V_tip          = R_tip*omega                    # blade_tip_speed         

                ## sound pressure for loading noise 
                #p_mL_GD = ((m*B*omega)/(2*np.sqrt(2)*np.pi*a*(S)))*np.sum((T_distribution*np.cos(theta) - (Q_distribution*a)/(omega*(R**2)))* 
                                                                       #jv(m*B,((m*B*omega*R*np.sin(theta))/a)))                
                ##p_mL_GD[np.isinf(p_mL_GD)] = 0
                
                ## sound pressure for thickness noise 
                #p_mT_GD = -((rho*((m*B*omega)**2)*B)/(3*np.sqrt(2)*np.pi*(S)))*np.sum(c*t*R*jv(m*B,((m*B*omega*R*np.sin(theta))/a)))
                ##p_mT_GD[np.isinf(p_mT_GD)] = 0
                ##p_mT_GD[np.isneginf(p_mT_GD)] = 0
                
                ## unweighted rotational sound pressure level
                #SPL_r_GD[h]        = 10*np.log10(N*((p_mL_GD**2 + p_mT_GD**2 )/p_ref**2))
                #p_pref_r_GD[h]     = 10**(SPL_r_GD[h]/10)  
                #SPL_r_GD_dBA[h]    = A_weighting(SPL_r_GD[h],f)                
                #p_pref_r_GD_dBA[h] = 10**(SPL_r_GD_dBA[h]/10)   
                
                ##------------------------------------------------------------------------
                ## Rotational SPL by Gutin & Deming SIMPLIFIED
                ##------------------------------------------------------------------------  
                
                ## sound pressure for loading noise 
                #p_mL_GD2 = ((m*B*omega)/(2*np.sqrt(2)*np.pi*a*(S)))*(T*np.cos(theta) - (Q*a)/(omega*(Re**2)))* jv(m*B,((m*B*omega*Re*np.sin(theta))/a))                
                #p_mL_GD2[np.isinf(p_mL_GD2)] = 0
                
                ## sound pressure for thickness noise 
                #p_mT_GD2 = -((rho*((m*B*omega)**2)*B)/(3*np.sqrt(2)* np.pi*(S))) * np.mean(c) * Re* np.mean(t) *jv(m*B,((m*B*omega*Re*np.sin(theta))/a))
                #p_mT_GD2[np.isinf(p_mT_GD2)] = 0
                #p_mT_GD2[np.isneginf(p_mT_GD2)] = 0
                
                ## unweighted rotational sound pressure level
                #SPL_r_GD2[h]        = 10*np.log10(N*((p_mL_GD2**2 + p_mT_GD2**2 )/p_ref**2))
                #p_pref_r_GD2[h]     = 10**(SPL_r_GD2[h]/10)  
                #SPL_r_GD_dBA2[h]    = A_weighting(SPL_r_GD2[h],f)                
                #p_pref_r_GD_dBA2[h] = 10**(SPL_r_GD_dBA2[h]/10)  
                
                ##------------------------------------------------------------------------
                ## Rotational SPL  by Barry & Magliozzi
                ##------------------------------------------------------------------------
                #Y            = np.sqrt(y**2 + z**2)                 # observer distance from propeller axis         
                #X            = np.sqrt(x**2 + z**2)                 # distance to observer from propeller plane (z)  
                #S0           = np.sqrt(z**2 + (1 - M**2)*(Y**2))    # amplitude radius                                
                #J_mB         = jv(m*B,((m*B*omega*R*Y)/a))          # Bessel function of order mB and argument x 
                #J_mB_plus_1  = jv((m*B+1),((m*B*omega*R*Y)/(a*S0))) # Bessel function of order mB and argument x 
                #J_mB_minus_1 = jv((m*B-1),((m*B*omega*R*Y)/(a*S0))) # Bessel function of order mB and argument x    
                #phi_t        = np.interp(R,R_old,phi_t_old)         # blade twist angle 
                #A_x          = 0.6853*c*t                       # airfoil cross-sectional area
                
                ## sound pressure for loading noise 
                #p_mL_BM = (1/(np.sqrt(2)*np.pi*S0))*np.sum( (R/(c*np.cos(phi_t)))*np.sin((m*B*c*np.cos(phi_t))/(2*R))* \
                                                         #((((M + X/S0)*omega)/(a*(1 - M**2)))*T_distribution - (1/R**2)*Q_distribution)* \
                                                         #(J_mB + (((1 - M**2)*Y*R)/(2*(S0**2)))*(J_mB_minus_1 - J_mB_plus_1)))               
                
                ##p_mL_BM[np.isinf(p_mL_BM)] = 0
                
                ## sound pressure for thickness noise 
                #p_mT_BM = -((rho*(m**2)*(omega**2)*(B**3) )/(2*np.sqrt(2)*np.pi*((1 - M**2)**2)))*(((S0 + M*X)**2)/(S0**3))* np.sum(A_x *(J_mB + (((1 - M**2)*Y*R)/(2*(S0**2)))*(J_mB_minus_1 - J_mB_plus_1))) 
                #p_mT_BM[np.isinf(p_mT_BM)] = 0
                #p_mT_BM[np.isneginf(p_mT_BM)] = 0
                
                ## unweighted rotational sound pressure level
                #SPL_r_BM[h]        = 10*np.log10(N*((p_mL_BM**2 + p_mT_BM**2 )/p_ref**2))
                #p_pref_r_BM[h]     = 10**(SPL_r_BM[h]/10)  
                #SPL_r_BM_dBA[h]    = A_weighting(SPL_r_BM[h],f)
                #p_pref_r_BM_dBA[h] = 10**(SPL_r_BM_dBA[h]/10)   
                
                ##------------------------------------------------------------------------
                ## Rotational SPL by Hanson
                ##------------------------------------------------------------------------
                #D             = 2*R_tip                             # propeller diameter
                #M_t           = V_tip/a                             # tip Mach number 
                #M_s           = np.sqrt(M**2 + (r**2)*(M_t**2))     # section relative Mach number 
                #r_t           = R_tip                               # propeller tip radius
                #phi           = np.arctan(z/y)            # tangential angle     np.arctan(z/y) 
                
                #print('phi')
                #print(phi/Units.degrees)
                
                #theta_r       = np.arccos(np.cos(theta)*np.sqrt(1 - (M**2)*(np.sin(theta))**2) + M*(np.sin(theta))**2 )      # theta angle in the retarded reference frame
                #theta_r_prime = np.arccos(np.cos(theta_r)*np.cos(alpha) + np.sin(theta_r)*np.sin(phi)*np.sin(alpha) )  #
                #phi_prime     = np.arccos((np.sin(theta_r)/np.sin(theta_r_prime))*np.cos(phi))                         # phi angle relative to propeller shaft axis                                                   
                #phi_s         = ((2*m*B*M_t)/(M_s*(1 - M*np.cos(theta_r))))*(MCA/D)                                   # phase lag due to sweep
                #J_mB          = jv(m*B,((m*B*r*omega*M_t*np.sin(theta_r_prime))/(1 - M*np.cos(theta_r))))              # Bessel function of order mB and argument x
                #S_r           = Y/(np.sin(theta_r))                                                                    # distance in retarded reference frame 
                #k_x           = ((2*m*B*c*M_t)/(M_s*(1 - M*np.cos(theta_r))))                                          # wave number 
                #psi_L         = np.zeros((n,1))
                #psi_V         = np.zeros((n,1))
                #for idx in range(n):
                    #if k_x[idx] == 0:
                        #psi_L[idx] = 2/3                                                                              # normalized loading souce transforms
                        #psi_V[idx] = (8/k_x[idx]**2)*((2/k_x[idx])*np.sin(0.5*k_x[idx])- np.cos(0.5*k_x[idx]))        # normalized thickness souce transforms
                    #else:                                                                                             
                        #psi_L[idx] = 1                                                                                # normalized loading souce transforms
                        #psi_V[idx] = (2/k_x[idx])*np.sin(0.5*k_x[idx])                                                # normalized thickness souce transforms           
                
                ## sound pressure for loading noise 
                #p_mL_H = ((1j*m*B*M_t*np.sin(theta_r)*np.exp(1j*m*B*((omega*S_r/a)+(phi_prime - np.pi/2))))/(2*np.sqrt(2)*np.pi*Y*r_t*(1 - M*np.cos(theta_r)))) \
                    #* np.sum(((np.cos(theta_r_prime)/(1 - M*np.cos(theta_r)))*T_distribution - (1/((r**2)*M_t*r_t))*Q_distribution)*np.exp(1j*phi_s)*J_mB*psi_L)
                                    
                #p_mL_H[np.isinf(p_mL_H)] = 0
                #p_mL_H = p_mL_H.real
                
                ## sound pressure for thickness noise 
                #p_mT_H = (-(rho*(a**2)*B*np.sin(theta_r)*np.exp(1j*m*B*((omega*S_r/a)+(phi_prime - np.pi/2))))/(4*np.sqrt(2)*np.pi*(Y/D)*(1 - M*np.cos(theta_r)))) \
                    #* np.sum((M_s**2)*(t/c)*np.exp(1j*phi_s)*J_mB*(k_x**2)*psi_V) 
                #p_mT_H[np.isinf(p_mT_H)] = 0  
                #p_mT_H = p_mT_H.real
                
                ## unweighted rotational sound pressure level
                #SPL_r_H[h]        = 10*np.log10(N*((p_mL_H**2 + p_mT_H**2 )/p_ref**2))  
                #p_pref_r_H[h]     = 10**(SPL_r_H[h]/10)  
                #SPL_r_H_dBA[h]    = A_weighting(SPL_r_H[h],f)
                #p_pref_r_H_dBA[h] = 10**(SPL_r_H_dBA[h]/10)   
                               
                
            ## -----------------------------------------------------------------------
            ## Vortex Noise (developed by Schlegel et. al)
            ## ----------------------------------------------------------------------- 
            #St       = 0.28                                        # Strouhal number             
            #t_avg    = np.mean(t)                                  # thickness
            #c_avg    = np.mean(c)                                  # average chord  
            #A_blade  = np.trapz(c_old, dx = (R_old[1] - R_old[0])) # area of blade
            #s        = (B*A_blade)/A                               # rotor solidity
            #K        = 6.1e-11    
            #K_2      = np.sqrt((0.7**2)*(15**2)*K*S**2)               
            
            ## unweighted overall vortex sound pressure level
            #SPL_v    = 20*np.log10((K_2)*(V_tip/(rho*S))*np.sqrt(((N*T)/s)*(T/A)))
            #SPL_v[np.isnan(SPL_v)] = 0
            
            ## estimation of A-Weighting for Vortex Noise  
            #Cl_bar      = (6*T)/(rho*s*V_tip*A)                                                # mean lift coefficient
            #blade_alpha = Cl_bar/(2*np.pi)                                                     # blade angle of attack at r/R = 0.7
            #h           = t_avg*np.cos(blade_alpha) + c_avg*np.sin(blade_alpha)                # projected blade thickness                   
            #f_peak      = (V_07*St)/h                                                          # V - blade velocity at a radial location of 0.7                
            #f_spectrum  = [0.5*f_peak, 1*f_peak , 2*f_peak , 4*f_peak , 8*f_peak , 16*f_peak]  # spectrum
            #SPL_weight  = [7.92 , 4.17 , 8.33 , 8.75 ,12.92 , 13.33]                           # SPL weight
            #SPL_v       = np.ones_like(SPL_weight)*SPL_v - SPL_weight                          # SPL correction
            #SPL_v_dBA   = np.zeros((len(SPL_v),1))
            #for idx in range(len(SPL_v)):
                #SPL_v_dBA[idx] = A_weighting(SPL_v[idx],f)
            
            #dim         = len(f_spectrum)
            #C           = np.zeros(dim) 
            #fr          = f_spectrum/f_peak                                                             # frequency ratio  
            #p_pref_v_dBA= np.zeros((len(SPL_v),1))
            
            ## A-Weighting for Vortex Noise  
            #for j in range(dim-1):
                #C[j]        = (SPL_v[j+1] - SPL_v[j])/(np.log10(fr[j+1]) - np.log10(fr[j])) 
                #C[j+1]      = SPL_v[j+1]- C[j]*np.log10(fr[j+1])   
                #p_pref_v_dBA[j] = (10** (0.1*C[j+1]))* (((fr[j+1]**(0.1*C[j] + 1 ))/(0.1*C[j] + 1 )) - ((fr[j]**(0.1*C[j] + 1 ))/(0.1*C[j] + 1 )) )
                    
            ## collecting unweighted pressure ratios 
            #total_p_pref_r_GD2.append(p_pref_r_GD2) 
            #total_p_pref_r_GD.append(p_pref_r_GD)  
            #total_p_pref_r_BM.append(p_pref_r_BM)  
            #total_p_pref_r_H.append(p_pref_r_H)    
            
            ## collecting weighted pressure ratios with vortex noise included
            #total_p_pref_GDv_dBA2.append(np.concatenate([p_pref_r_GD_dBA2,p_pref_v_dBA])) # Gutin & Deming rotational noise with Schlegel vortex noise
            #total_p_pref_GDv_dBA.append(np.concatenate([p_pref_r_GD_dBA,p_pref_v_dBA])) # Gutin & Deming rotational noise with Schlegel vortex noise
            #total_p_pref_BMv_dBA.append(np.concatenate([p_pref_r_BM_dBA,p_pref_v_dBA])) # Barry & Magliozzi rotational noise with Schlegel vortex noise
            #total_p_pref_Hv_dBA.append(np.concatenate([p_pref_r_H_dBA,p_pref_v_dBA])) # Hanson rotational noise with Schlegel vortex noise        
        
        ## Rotational SPL (Unweighted)   
        #SPL_DG_unweighted2[i,0] = decibel_arithmetic(total_p_pref_r_GD2)
        #SPL_DG_unweighted[i,0] = decibel_arithmetic(total_p_pref_r_GD)
        #SPL_BM_unweighted[i,0] = decibel_arithmetic(total_p_pref_r_BM)
        #SPL_H_unweighted[i,0]  = decibel_arithmetic(total_p_pref_r_H)            
        
        ## A- Weighted Rotational and Vortex SPL
        #SPL_GDv_dBA2[i,0] = decibel_arithmetic( total_p_pref_GDv_dBA2)
        #SPL_GDv_dBA[i,0] = decibel_arithmetic( total_p_pref_GDv_dBA)
        #SPL_BMv_dBA[i,0] = decibel_arithmetic(total_p_pref_BMv_dBA)
        #SPL_Hv_dBA[i,0]  = decibel_arithmetic(total_p_pref_Hv_dBA)
    
    #return   SPL_DG_unweighted2 , SPL_DG_unweighted , SPL_BM_unweighted , SPL_H_unweighted , SPL_GDv_dBA2 , SPL_GDv_dBA  , SPL_BMv_dBA , SPL_Hv_dBA
## -----------------------------------------------------------------------
## Decibel Arithmetic
## -----------------------------------------------------------------------
#def decibel_arithmetic(p_pref_total):
    #SPL_total = 10*np.log10(np.sum(p_pref_total))
    #return SPL_total

## -----------------------------------------------------------------------
## Rotational Noise A-Weight
## -----------------------------------------------------------------------
#def  A_weighting(SPL,f):
    #Ra_f       = ((12200**2)*(f**4))/ (((f**2)+(20.6**2)) * ((f**2)+(12200**2)) * (((f**2) + 107.7**2)**0.5)* (((f**2)+ 737.9**2)**0.5)) 
    #A_f        =  2.0  + 20*np.log10(Ra_f) 
    #SPL_dBA = SPL + A_f 
    #return SPL_dBA