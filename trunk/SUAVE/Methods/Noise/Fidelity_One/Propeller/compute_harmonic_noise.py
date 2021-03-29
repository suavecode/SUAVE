## @ingroupMethods-Noise-Fidelity_One-Propeller
# compute_harmonic_noise.py
#
# Created:  Mar 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE 
from SUAVE.Core import Units , Data 
import numpy as np
from scipy.special import jv 
 
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.dbA_noise  import A_weighting  
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools            import SPL_harmonic_to_third_octave

# ----------------------------------------------------------------------
# Harmonic Noise Domain Broadband Noise Computation
# ----------------------------------------------------------------------
## @ingroupMethods-Noise-Fidelity_One-Propeller
def compute_harmonic_noise(i,num_h,p_idx,harmonics ,num_f,freestream,angle_of_attack,position_vector,
                           velocity_vector, mic_loc,propeller,auc_opts,settings,res):
    '''This computes the  harmonic noise (i.e. thickness and loading noise) of a propeller or rotor
    in the frequency domain
    
    Assumptions:
    Compactness of thrust and torque along blade radius from root to tip

    Source:
    1) Hanson, Donald B. "Helicoidal surface theory for harmonic noise of propellers in the far field."
    AIAA Journal 18.10 (1980): 1213-1220.
    
    2) Hubbard, Harvey H., ed. Aeroacoustics of flight vehicles: theory and practice. Vol. 1.
    NASA Office of Management, Scientific and Technical Information Program, 1991.
    
    
    Inputs:
        i                             - control point 
        num_h                         - number of harmonics
        p_idx                         - index of propeller/rotor
        harmonics                     - harmomics 
        num_f                         - number of freqencies 
        freestream                    - freestream data structure
        angle_of_attack               - aircraft angle of attack
        position_vector               - position vector of aircraft
        velocity_vector               - velocity vector of aircraft
        mic_loc                       - microhone location 
        propeller                     - propeller class data structure
        auc_opts                      - data structure of acoustic data
        settings                      - accoustic settings 
        
        res.      
            SPL_prop_tonal_spectrum   - SPL of Frequency Spectrum
            SPL_bpfs_spectrum         - Blade Passing Frequency Spectrum
            SPL_prop_h_spectrum       - Rotational Noise Frequency Spectrum in 1/3 octave spectrum 
            SPL_prop_h_dBA_spectrum   - dBA-WeightedRotational Noise Frequency Spectrum in 1/3 octave spectrum 
    
    Outputs
       *acoustic data is stored in passed in data structures*
            
    Properties Used:
        N/A   
    '''    
    # ----------------------------------------------------------------------------------
    # Rotational Noise  Thickness and Loading Noise
    # ----------------------------------------------------------------------------------  
    for h in range(num_h):
        m              = harmonics[h]                      # harmonic number 
        p_ref          = 2E-5                              # referece atmospheric pressure
        a              = freestream.speed_of_sound[i][0]   # speed of sound
        rho            = freestream.density[i][0]          # air density 
        AoA            = angle_of_attack[i][0]             # vehicle angle of attack  
        thrust_angle   = auc_opts.thrust_angle             # propeller thrust angle
        alpha          = AoA + thrust_angle    
        x              = position_vector[0] 
        y              = position_vector[1]
        z              = position_vector[2]
        Vx             = velocity_vector[i][0]             # x velocity of propeller  
        Vy             = velocity_vector[i][1]             # y velocity of propeller 
        Vz             = velocity_vector[i][2]             # z velocity of propeller 
        B              = propeller.number_of_blades        # number of propeller blades
        omega          = auc_opts.omega[i]                 # angular velocity       
        dT_dr          = auc_opts.blade_dT_dr[i]           # nondimensionalized differential thrust distribution 
        dQ_dr          = auc_opts.blade_dQ_dr[i]           # nondimensionalized differential torque distribution
        R              = propeller.radius_distribution     # radial location     
        c              = propeller.chord_distribution      # blade chord    
        R_tip          = propeller.tip_radius 
        t_c            = propeller.thickness_to_chord      # thickness to chord ratio
        MCA            = propeller.mid_chord_aligment      # Mid Chord Alighment 
    
        res.f[h]       = B*omega*m/(2*np.pi)   
        n              = len(R)  
        D              = 2*R[-1]                           # propeller diameter    
        r              = R/R[-1]                           # non dimensional radius distribution   
        S              = np.sqrt(x**2 + y**2 + z**2)       # distance between rotor and the observer    
        theta          = np.arccos(x/S)                    
        Y              = np.sqrt(y**2 + z**2)              # observer distance from propeller axis          
        V              = np.sqrt(Vx**2 + Vy**2 + Vz**2)    # velocity magnitude
        M_x            = V/a                               
        V_tip          = R_tip*omega                       # blade_tip_speed 
        M_t            = V_tip/a                           # tip Mach number 
        M_r            = np.sqrt(M_x**2 + (r**2)*(M_t**2))   # section relative Mach number     
        B_D            = c/D                               
        phi            = np.arctan(z/y)                    # tangential angle   
        
        # retarted  theta angle in the retarded reference frame
        theta_r        = np.arccos(np.cos(theta)*np.sqrt(1 - (M_x**2)*(np.sin(theta))**2) + M_x*(np.sin(theta))**2 )   
        theta_r_prime  = np.arccos(np.cos(theta_r)*np.cos(alpha) + np.sin(theta_r)*np.sin(phi)*np.sin(alpha) )
        
        # phi angle relative to propeller shaft axis
        phi_prime      = np.arccos((np.sin(theta_r)/np.sin(theta_r_prime))*np.cos(phi))                                                                                 
        k_x            = ((2*m*B*B_D*M_t)/(M_r*(1 - M_x*np.cos(theta_r))))      # wave number
        k_y            = ((2*m*B*B_D)/(M_r*r))*((M_x - (M_r**2)*np.cos(theta_r))/(1 - M_x*np.cos(theta_r)))
        phi_s          = ((2*m*B*M_t)/(M_r*(1 - M_x*np.cos(theta_r))))*(MCA/D)
        S_r            = Y/(np.sin(theta_r))                                # distance in retarded reference frame   
        Jmb            = jv(m*B,((m*B*r*M_t*np.sin(theta_r_prime))/(1 - M_x*np.cos(theta_r))))  
        psi_L          = np.zeros(n)
        psi_V          = np.zeros(n)
       
        # normalized thickness  and loading shape functions                
        psi_V[0]       = 2/3   
        psi_L[0]       = 1     
        psi_V[1:]      = (8/(k_x[1:]**2))*((2/k_x[1:])*np.sin(0.5*k_x[1:]) - np.cos(0.5*k_x[1:]))    
        psi_L[1:]      = (2/k_x[1:])*np.sin(0.5*k_x[1:])                  
         
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
    
        # unweighted harmonic sound pressure level 
        res.SPL_r[p_idx,h]        = 20*np.log10((np.linalg.norm(p_mL_H + p_mT_H))/p_ref) 
        res.p_pref_r[p_idx,h]     = 10**(res.SPL_r[p_idx,h]/10)   
        res.SPL_r_dBA[p_idx,h]    = A_weighting(res.SPL_r[p_idx,h],res.f[h]) 
        res.p_pref_r_dBA[p_idx,h] = 10**(res.SPL_r_dBA[p_idx,h]/10)    
        
    # convert to 1/3 octave spectrum 
    res.SPL_prop_h_spectrum[i,p_idx,:]     = SPL_harmonic_to_third_octave(res.SPL_r[p_idx,:],res.f,settings)         
    res.SPL_prop_h_dBA_spectrum[i,p_idx,:] = SPL_harmonic_to_third_octave(res.SPL_r_dBA[p_idx,:],res.f,settings)     
  
    # Rotational(periodic/tonal)   
    res.SPL_prop_tonal_spectrum[i,p_idx,:] = 10*np.log10( 10**(res.SPL_prop_h_spectrum[i,p_idx,:]/10))
 
    return  
 
