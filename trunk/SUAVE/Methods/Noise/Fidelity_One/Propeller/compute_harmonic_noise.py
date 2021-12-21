## @ingroupMethods-Noise-Fidelity_One-Propeller
# compute_harmonic_noise.py
#
# Created:  Mar 2021, M. Clarke
# Modified: Jul 2021, E. Botero

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
def compute_harmonic_noise(harmonics,freestream,angle_of_attack,position_vector,
                           velocity_vector,network,auc_opts,settings,res,source):
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
        harmonics                     - harmomics                                                                  [Unitless]
        freestream                    - freestream data structure                                                  [m/s]
        angle_of_attack               - aircraft angle of attack                                                   [rad]
        position_vector               - position vector of aircraft                                                [m]
        velocity_vector               - velocity vector of aircraft                                                [m/s]
        mic_loc                       - microhone location                                                         [None]
        propeller                     - propeller class data structure                                             [None]
        auc_opts                      - data structure of acoustic data                                            [None]
        settings                      - accoustic settings                                                     
    
    Outputs 
        res.                                    *acoustic data is stored and passed in data structures*                                                                            
            SPL_prop_harmonic_bpf_spectrum       - harmonic noise in blade passing frequency spectrum              [dB]
            SPL_prop_harmonic_bpf_spectrum_dBA   - dBA-Weighted harmonic noise in blade passing frequency spectrum [dbA]                  
            SPL_prop_harmonic_1_3_spectrum       - harmonic noise in 1/3 octave spectrum                           [dB]
            SPL_prop_harmonic_1_3_spectrum_dBA   - dBA-Weighted harmonic noise in 1/3 octave spectrum              [dBA] 
            p_pref_harmonic                      - pressure ratio of harmonic noise                                [Unitless]
            p_pref_harmonic_dBA                  - pressure ratio of dBA-weighted harmonic noise                   [Unitless]
 
            
    Properties Used:
        N/A   
    '''     
    num_h           = len(harmonics)     
    num_cpt         = len(angle_of_attack)
    num_mic         = len(position_vector[0,:,0,1])
    num_prop        = len(position_vector[0,0,:,1])
    num_h           = len(harmonics) 
    num_azi         = auc_opts.number_azimuthal_stations

    if source == 'lift_rotors':  
        propellers      = network.lift_rotors 
    else:
        propellers      = network.propellers
        
    propeller       = propellers[list(propellers.keys())[0]] 
    num_r           = len(propeller.radius_distribution)  
    precision       = settings.floating_point_precision    
    
    # ----------------------------------------------------------------------------------
    # Rotational Noise  Thickness and Loading Noise
    # ----------------------------------------------------------------------------------  
    # [control point ,microphones, propellers, radial distribution, harmonics]  
    m              = vectorize(harmonics,num_cpt,num_h,num_r,num_prop,num_mic,num_azi,precision,vectorize_method = 1)                     # harmonic number 
    m_1d           = harmonics                                                                                          
    p_ref          = 2E-5                                                                                               # referece atmospheric pressure
    a              = vectorize(freestream.speed_of_sound[:,0],num_cpt,num_h,num_r,num_prop,num_mic,num_azi,precision,vectorize_method = 2)# speed of sound
    rho            = vectorize(freestream.density[:,0],num_cpt,num_h,num_r,num_prop,num_mic,num_azi,precision,vectorize_method = 2)       # air density 
    AoA            = angle_of_attack[:,0]                                                                               # vehicle angle of attack  
    thrust_angle   = propeller.orientation_euler_angles[1]                                                                       # propeller thrust angle
    alpha          = vectorize((AoA + thrust_angle),num_cpt,num_h,num_r,num_prop,num_mic,num_azi,precision,vectorize_method = 2)            
    x              = vectorize(position_vector[:,:,:,0],num_cpt,num_h,num_r,num_prop,num_mic,num_azi,precision,vectorize_method = 3)      # x component of position vector of propeller to microphone 
    y              = vectorize(position_vector[:,:,:,1],num_cpt,num_h,num_r,num_prop,num_mic,num_azi,precision,vectorize_method = 3)      # y component of position vector of propeller to microphone
    z              = vectorize(position_vector[:,:,:,2],num_cpt,num_h,num_r,num_prop,num_mic,num_azi,precision,vectorize_method = 3)      # z component of position vector of propeller to microphone
    Vx             = vectorize(velocity_vector[:,0],num_cpt,num_h,num_r,num_prop,num_mic,num_azi,precision,vectorize_method = 2)          # x velocity of propeller  
    Vy             = vectorize(velocity_vector[:,1],num_cpt,num_h,num_r,num_prop,num_mic,num_azi,precision,vectorize_method = 2)          # y velocity of propeller 
    Vz             = vectorize(velocity_vector[:,2],num_cpt,num_h,num_r,num_prop,num_mic,num_azi,precision,vectorize_method = 2)          # z velocity of propeller 
    B              = propeller.number_of_blades                                                                         # number of propeller blades
    omega          = vectorize(auc_opts.omega[:,0],num_cpt,num_h,num_r,num_prop,num_mic,num_azi,precision,vectorize_method = 2)           # angular velocity       
    dT_dr          = vectorize(auc_opts.blade_dT_dr,num_cpt,num_h,num_r,num_prop,num_mic,num_azi,precision,vectorize_method = 4)          # nondimensionalized differential thrust distribution 
    dQ_dr          = vectorize(auc_opts.blade_dQ_dr,num_cpt,num_h,num_r,num_prop,num_mic,num_azi,precision,vectorize_method = 4)          # nondimensionalized differential torque distribution
    R              = vectorize(propeller.radius_distribution,num_cpt,num_h,num_r,num_prop,num_mic,num_azi,precision,vectorize_method = 5) # radial location     
    c              = vectorize(propeller.chord_distribution,num_cpt,num_h,num_r,num_prop,num_mic,num_azi,precision,vectorize_method = 5)  # blade chord    
    R_tip          = propeller.tip_radius                                                     
    t_c            = vectorize(propeller.thickness_to_chord,num_cpt,num_h,num_r,num_prop,num_mic,num_azi,precision,vectorize_method = 5)  # thickness to chord ratio
    MCA            = vectorize(propeller.mid_chord_alignment,num_cpt,num_h,num_r,num_prop,num_mic,num_azi,precision,vectorize_method = 5) # Mid Chord Alighment  
    res.f          = B*omega*m/(2*np.pi) 
    D              = 2*R[0,0,0,-1,:]                                                                                    # propeller diameter    
    r              = R/R[0,0,0,-1,:]                                                                                    # non dimensional radius distribution  
    S              = np.sqrt(x**2 + y**2 + z**2)                                                                        # distance between rotor and the observer    
    theta          = np.arccos(x/S)                                                            
    Y              = np.sqrt(y**2 + z**2)                                                                               # observer distance from propeller axis          
    V              = np.sqrt(Vx**2 + Vy**2 + Vz**2)                                                                     # velocity magnitude
    M_x            = V/a                                                                                                
    V_tip          = R_tip*omega                                                                                        # blade_tip_speed 
    M_t            = V_tip/a                                                                                            # tip Mach number 
    M_r            = np.sqrt(M_x**2 + (r**2)*(M_t**2))                                                                  # section relative Mach number     
    B_D            = c/D                                                                                                
    phi            = np.arctan(z/y)                                                                                     # tangential angle   

    # retarted  theta angle in the retarded reference frame
    theta_r        = np.arccos(np.cos(theta)*np.sqrt(1 - (M_x**2)*(np.sin(theta))**2) + M_x*(np.sin(theta))**2 )   
    theta_r_prime  = np.arccos(np.cos(theta_r)*np.cos(alpha) + np.sin(theta_r)*np.sin(phi)*np.sin(alpha) )

    # phi angle relative to propeller shaft axis
    psi_L          = np.zeros((num_cpt,num_mic,num_prop,num_r,num_h)).astype(precision) 
    psi_V          = np.zeros((num_cpt,num_mic,num_prop,num_r,num_h)).astype(precision) 

    # normalized thickness  and loading shape functions                
    k_x               = ((2*m*B*B_D*M_t)/(M_r*(1 - M_x*np.cos(theta_r))))      # wave number 
    psi_V[:,:,:,0,:]  = 2/3   
    psi_L[:,:,:,0,:]  = 1     
    psi_V[:,:,:,1:,:] = (8/(k_x[:,:,:,1:,:]**2))*((2/k_x[:,:,:,1:,:])*np.sin(0.5*k_x[:,:,:,1:,:]) - np.cos(0.5*k_x[:,:,:,1:,:]))    
    psi_L[:,:,:,1:,:] = (2/k_x[:,:,:,1:,:])*np.sin(0.5*k_x[:,:,:,1:,:])                  

    # sound pressure for thickness noise   
    Jmb               = jv(m*B,((m*B*r*M_t*np.sin(theta_r_prime))/(1 - M_x*np.cos(theta_r))))  
    phi_s             = ((2*m*B*M_t)/(M_r*(1 - M_x*np.cos(theta_r))))*(MCA/D)
    phi_prime         = np.arccos((np.sin(theta_r)/np.sin(theta_r_prime))*np.cos(phi))      
    S_r               = Y/(np.sin(theta_r))                                # distance in retarded reference frame                                                                             
    exponent_fraction = np.exp(1j*m_1d*B*((omega*S_r/a) +  phi_prime - np.pi/2))/(1 - M_x*np.cos(theta_r))
    p_mT_H_integral   = -((M_r**2)*(t_c)*np.exp(1j*phi_s)*Jmb*(k_x**2)*psi_V ) * ((rho*(a**2)*B*np.sin(theta_r))/(4*np.sqrt(2)*np.pi*(Y/D)))* exponent_fraction
    p_mT_H            = np.trapz(p_mT_H_integral,x = r[0,0,0,:,0], axis =3) 
    
    p_mT_H_abs        = abs(p_mT_H)             
    p_mL_H_integral   = (((np.cos(theta_r_prime)/(1 - M_x*np.cos(theta_r)))*dT_dr - (1/((r**2)*M_t*R_tip))*dQ_dr)
                                  * np.exp(1j*phi_s)*Jmb * psi_L)*(m_1d*B*M_t*np.sin(theta_r)/ (2*np.sqrt(2)*np.pi*Y*R_tip)) *exponent_fraction
    p_mL_H            = np.trapz(p_mL_H_integral,x = r[0,0,0,:,0], axis = 3 ) 
    p_mL_H_abs        =  abs(p_mL_H)  
    
    # sound pressure levels  
    res.SPL_prop_harmonic_bpf_spectrum     = 20*np.log10((abs(p_mL_H_abs + p_mT_H_abs))/p_ref) 
    res.SPL_prop_harmonic_bpf_spectrum_dBA = A_weighting(res.SPL_prop_harmonic_bpf_spectrum,res.f[:,:,:,0,:]) 
    res.p_pref_harmonic                    = 10**(res.SPL_prop_harmonic_bpf_spectrum/10)   
    res.p_pref_harmonic_dBA                = 10**(res.SPL_prop_harmonic_bpf_spectrum_dBA/10) 
    res.SPL_prop_harmonic_1_3_spectrum     = SPL_harmonic_to_third_octave(res.SPL_prop_harmonic_bpf_spectrum,res.f[:,0,0,0,:],settings)         
    res.SPL_prop_harmonic_1_3_spectrum_dBA = SPL_harmonic_to_third_octave(res.SPL_prop_harmonic_bpf_spectrum_dBA,res.f[:,0,0,0,:],settings)  

    # compute acoustic waveforms around azimuth of propeller 
    p_mT_H_azi = vectorize(p_mT_H,num_cpt,num_h,num_r,num_prop,num_mic,num_azi,precision,vectorize_method = 6)
    p_mL_H_azi = vectorize(p_mL_H,num_cpt,num_h,num_r,num_prop,num_mic,num_azi,precision,vectorize_method = 6)
    m_azi      = vectorize(m[:,:,:,0,:],num_cpt,num_h,num_r,num_prop,num_mic,num_azi,precision,vectorize_method = 6)
    omega_azi  = vectorize(omega[:,:,:,0,:],num_cpt,num_h,num_r,num_prop,num_mic,num_azi,precision,vectorize_method = 6) 
    time       = vectorize(auc_opts.omega,num_cpt,num_h,num_r,num_prop,num_mic,num_azi,precision,vectorize_method = 7)
    p_t        = np.sum((p_mT_H_azi + p_mL_H_azi)*np.exp(-1j*m_azi*B*omega_azi*time),axis = 4)
    p_t_abs    = abs(p_t)

    res.p_harmonic     = p_t_abs 
    res.azimuthal_time = time
    
    return  

## @ingroupMethods-Noise-Fidelity_One-Propeller 
def vectorize(X0,num_cpt,num_h,num_r,num_prop,num_mic,num_azi,precision,vectorize_method): 
    ''' This vectorizes a variable for acoustic computation 
    
    Assumptions:
        None

    Source:
        None

    Inputs:                                                       
        X0               - input argument                           [None]
        num_cpt          - number of control points                 [Unitless]
        num_h            - number of harmonics                      [Unitless]
        num_r            - number of radial stations on rotor       [Unitless]
        num_prop         - number of propellers/lift_rotors         [Unitless]
        num_mic          - number of microphones                    [Unitless]
        num_azi          - number of aximuthal stations on rotor    [Unitless]
        vectorize_method - method of vectorizing data               [Unitless]

    Outputs: 
        vectorized_x - vectorized output

    Properties Used:
        N/A 
    '''
    

    X  = np.array(X0, dtype=precision)
    
    if   vectorize_method == 1:
        vectorized_x = np.repeat(np.repeat(np.repeat(np.repeat(np.atleast_2d(X),num_r ,axis = 0)[np.newaxis,:,:],
                            num_prop, axis = 0)[np.newaxis,:,:,:],num_mic, axis = 0)[np.newaxis,:,:,:,:],num_cpt, axis = 0)        
    
    elif vectorize_method == 2:
        vectorized_x = np.repeat(np.repeat(np.repeat(np.repeat(np.atleast_2d(X).T,num_mic, axis = 1)[:,:,np.newaxis],
                       num_prop, axis = 2)[:,:,:,np.newaxis] ,num_r , axis = 3)[:,:,:,:,np.newaxis],num_h , axis = 4)        
    
    elif vectorize_method == 3: 
        vectorized_x = np.repeat(np.repeat(X[:,:,:,np.newaxis] ,num_r , axis = 3)[:,:,:,:,np.newaxis],num_h , axis = 4)
    
    elif vectorize_method == 4: 
        vectorized_x = np.repeat(np.repeat(np.repeat(X[:,np.newaxis,:],num_mic, axis = 1)[:,:,np.newaxis,:],
                       num_prop, axis = 2)[:,:,:,:,np.newaxis],num_h , axis = 4)        
    
    elif vectorize_method == 5: 
        vectorized_x = np.repeat(np.repeat(np.repeat(np.repeat(np.atleast_2d(X).T,num_h , axis = 1)[np.newaxis,:,:],
                       num_prop, axis = 0)[np.newaxis,:,:,:],num_mic, axis = 0)[np.newaxis,:,:,:,:],num_cpt, axis = 0)
        
    elif vectorize_method == 6: 
        vectorized_x = np.repeat(X[:,:,:,np.newaxis,:],num_azi, axis = 3)
   
    elif vectorize_method == 7: 
        non_dim_time = np.repeat(np.atleast_2d(np.linspace(0,1,num_azi)),num_cpt, axis = 0)
        period       = np.repeat((2*np.pi)/X,num_azi, axis = 1)
        time         = non_dim_time*period
        vectorized_x = np.repeat(np.repeat(np.repeat(time[:,np.newaxis,:],num_mic, axis = 1)[:,:,np.newaxis,:],
                       num_prop, axis = 2)[:,:,:,:,np.newaxis],num_h, axis = 4)
    return vectorized_x