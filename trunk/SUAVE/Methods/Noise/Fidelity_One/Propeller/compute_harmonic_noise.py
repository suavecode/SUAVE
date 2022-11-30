## @ingroup Methods-Noise-Fidelity_One-Propeller
# compute_harmonic_noise.py
#
# Created:  Mar 2021, M. Clarke
# Modified: Jul 2021, E. Botero
#           Feb 2022, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
from jax import  jit
import jax.numpy as jnp 
from SUAVE.Core.Utilities import jjv

from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.dbA_noise  import A_weighting  
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools            import convert_to_one_third_octave_band

# ----------------------------------------------------------------------
# Harmonic Noise Domain Broadband Noise Computation
# ----------------------------------------------------------------------
## @ingroup Methods-Noise-Fidelity_One-Propeller
@jit
def compute_harmonic_noise(harmonics,freestream,angle_of_attack,position_vector,
                           velocity_vector,rotors,aeroacoustic_data,settings,res):
    '''This computes the  harmonic noise (i.e. thickness and loading noise) of a rotor or rotor
    in the frequency domain

    Assumptions:
    Compactness of thrust and torque along blade radius from root to tip

    Source:
    1) Hanson, Donald B. "Helicoidal surface theory for harmonic noise of rotors in the far field."
    AIAA Journal 18.10 (1980): 1213-1220.

    2) Hubbard, Harvey H., ed. Aeroacoustics of flight vehicles: theory and practice. Vol. 1.
    NASA Office of Management, Scientific and Technical Information Program, 1991.


    Inputs: 
        harmonics                     - harmomics                                                                  [Unitless]
        freestream                    - freestream data structure                                                  [m/s]
        angle_of_attack               - aircraft angle of attack                                                   [rad]
        position_vector               - position vector of aircraft                                                [m]
        velocity_vector               - velocity vector of aircraft                                                [m/s] 
        rotors                        - data structure of rotors                                                   [None]
        aeroacoustic_data             - data structure of acoustic data                                            [None]
        settings                      - accoustic settings                                                         [None] 
        res                           - results data structure                                                     [None] 

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
    num_h        = len(harmonics)     
    num_cpt      = len(angle_of_attack)
    num_mic      = len(position_vector[0,:,0,0])
    num_rot      = len(position_vector[0,0,:,0])  
    rotor        = rotors[list(rotors.keys())[0]] 
    num_r        = len(rotor.radius_distribution) 
    body2thrust  = rotor.body_to_prop_vel()[0]
    
    # ----------------------------------------------------------------------------------
    # Rotational Noise  Thickness and Loading Noise
    # ----------------------------------------------------------------------------------  
    # [control point ,microphones, rotors, radial distribution, harmonics]  
    m              = jnp.tile(harmonics[None,None,None,None,:],(num_cpt,num_mic,num_rot,num_r,1))      # harmonic number 
    m_1d           = harmonics                                                                                       
    p_ref          = 2E-5                                                                                        # referece atmospheric pressure
    a              = jnp.tile(freestream.speed_of_sound[:,:,None,None,None],(1,num_mic,num_rot,num_r,num_h))     # speed of sound
    rho            = jnp.tile(freestream.density[:,:,None,None,None],(1,num_mic,num_rot,num_r,num_h))             # air density   
    alpha          = jnp.tile((angle_of_attack + jnp.arccos(body2thrust[0,0]))[:,:,None,None,None],(1,num_mic,num_rot,num_r,num_h))           
    x              = jnp.tile(position_vector[:,:,:,0][:,:,:,None,None],(1,1,1,num_r,num_h))                     # x component of position vector of rotor to microphone 
    y              = jnp.tile(position_vector[:,:,:,1][:,:,:,None,None],(1,1,1,num_r,num_h))                     # y component of position vector of rotor to microphone
    z              = jnp.tile(position_vector[:,:,:,2][:,:,:,None,None],(1,1,1,num_r,num_h))                     # z component of position vector of rotor to microphone
    Vx             = jnp.tile(velocity_vector[:,0][:,None,None,None,None],(1,num_mic,num_rot,num_r,num_h))        # x velocity of rotor  
    Vy             = jnp.tile(velocity_vector[:,1][:,None,None,None,None],(1,num_mic,num_rot,num_r,num_h))        # y velocity of rotor 
    Vz             = jnp.tile(velocity_vector[:,2][:,None,None,None,None],(1,num_mic,num_rot,num_r,num_h))        # z velocity of rotor 
    B              = rotor.number_of_blades                                                                      # number of rotor blades
    omega          = jnp.tile(aeroacoustic_data.omega[:,:,None,None,None],(1,num_mic,num_rot,num_r,num_h))       # angular velocity       
    dT_dr          = jnp.tile(aeroacoustic_data.blade_dT_dr[:,None,None,:,None],(1,num_mic,num_rot,1,num_h))     # nondimensionalized differential thrust distribution 
    dQ_dr          = jnp.tile(aeroacoustic_data.blade_dQ_dr[:,None,None,:,None],(1,num_mic,num_rot,1,num_h))     # nondimensionalized differential torque distribution
    R              = jnp.tile(rotor.radius_distribution[None,None,None,:,None],(num_cpt,num_mic,num_rot,1,num_h))# radial location     
    c              = jnp.tile(rotor.chord_distribution[None,None,None,:,None],(num_cpt,num_mic,num_rot,1,num_h)) # blade chord    
    R_tip          = rotor.tip_radius                                                     
    t_c            = jnp.tile(rotor.thickness_to_chord[None,None,None,:,None],(num_cpt,num_mic,num_rot,1,num_h)) # thickness to chord ratio
    MCA            = jnp.tile(rotor.mid_chord_alignment[None,None,None,:,None],(num_cpt,num_mic,num_rot,1,num_h))# Mid Chord Alighment  
    
    
    res.f          = B*omega*m/(2*jnp.pi) 
    D              = 2*R[0,0,0,-1,:]                                                                             # rotor diameter    
    r              = R/R[0,0,0,-1,:]                                                                             # non dimensional radius distribution  
    S              = jnp.sqrt(x**2 + y**2 + z**2)                                                                 # distance between rotor and the observer    
    theta          = jnp.arccos(x/S)                                                            
    Y              = jnp.sqrt(y**2 + z**2)                                                                        # observer distance from rotor axis          
    V              = jnp.sqrt(Vx**2 + Vy**2 + Vz**2)                                                              # velocity magnitude
    M_x            = V/a                                                                                         
    V_tip          = R_tip*omega                                                                                 # blade_tip_speed 
    M_t            = V_tip/a                                                                                     # tip Mach number 
    M_r            = jnp.sqrt(M_x**2 + (r**2)*(M_t**2))                                                           # section relative Mach number     
    B_D            = c/D                                                                                         
    phi            = jnp.arctan(z/y)                                                                              # tangential angle   

    # retarted  theta angle in the retarded reference frame
    theta_r        = jnp.arccos(jnp.cos(theta)*jnp.sqrt(1 - (M_x**2)*(jnp.sin(theta))**2) + M_x*(jnp.sin(theta))**2 )   
    theta_r_prime  = jnp.arccos(jnp.cos(theta_r)*jnp.cos(alpha) + jnp.sin(theta_r)*jnp.sin(phi)*jnp.sin(alpha) )

    # initialize thickness and loading noise matrices
    psi_L          = jnp.zeros((num_cpt,num_mic,num_rot,num_r,num_h))
    psi_V          = jnp.zeros((num_cpt,num_mic,num_rot,num_r,num_h))

    # normalized thickness  and loading shape functions                
    k_x               = ((2*m*B*B_D*M_t)/(M_r*(1 - M_x*jnp.cos(theta_r))))      # wave number 
    psi_V             = psi_V.at[:,:,:,0,:].set(2/3)
    psi_L             = psi_L.at[:,:,:,0,:].set(1)     
    psi_V             = psi_V.at[:,:,:,1:,:].set((8/(k_x[:,:,:,1:,:]**2))*((2/k_x[:,:,:,1:,:])*jnp.sin(0.5*k_x[:,:,:,1:,:]) - jnp.cos(0.5*k_x[:,:,:,1:,:])))    
    psi_L             = psi_L.at[:,:,:,1:,:].set((2/k_x[:,:,:,1:,:])*jnp.sin(0.5*k_x[:,:,:,1:,:]))                  

    # sound pressure for thickness noise   
    Jmb               = jjv(m*B,((m*B*r*M_t*jnp.sin(theta_r_prime))/(1 - M_x*jnp.cos(theta_r))))   
    phi_s             = ((2*m*B*M_t)/(M_r*(1 - M_x*jnp.cos(theta_r))))*(MCA/D)
    phi_prime_var     = (jnp.sin(theta_r)/jnp.sin(theta_r_prime))*jnp.cos(phi) 
    #pt_ids            = jnp.where(phi_prime_var>1.0) 
    #phi_prime_var     = phi_prime_var.at[pt_ids].set(0) 
    phi_prime_var     = jnp.where(phi_prime_var>1.0,0,phi_prime_var) 
    phi_prime         = jnp.arccos(phi_prime_var)      
    S_r               = Y/(jnp.sin(theta_r))                                # distance in retarded reference frame                                                                             
    exponent_fraction = jnp.exp(1j*m_1d*B*((omega*S_r/a) +  phi_prime - jnp.pi/2))/(1 - M_x*jnp.cos(theta_r))
    p_mT_H_integral   = -((M_r**2)*(t_c)*jnp.exp(1j*phi_s)*Jmb*(k_x**2)*psi_V ) * ((rho*(a**2)*B*jnp.sin(theta_r))/(4*jnp.sqrt(2)*jnp.pi*(Y/D)))* exponent_fraction
    p_mT_H            = jnp.trapz(p_mT_H_integral,x = r[0,0,0,:,0], axis =3) 

    p_mT_H_abs        = abs(p_mT_H)             
    p_mL_H_integral   = (((jnp.cos(theta_r_prime)/(1 - M_x*jnp.cos(theta_r)))*dT_dr - (1/((r**2)*M_t*R_tip))*dQ_dr)
                         * jnp.exp(1j*phi_s)*Jmb * psi_L)*(m_1d*B*M_t*jnp.sin(theta_r)/ (2*jnp.sqrt(2)*jnp.pi*Y*R_tip)) *exponent_fraction
    p_mL_H            = jnp.trapz(p_mL_H_integral,x = r[0,0,0,:,0], axis = 3 ) 
    p_mL_H_abs        =  abs(p_mL_H)  

    # sound pressure levels  
    res.SPL_prop_harmonic_bpf_spectrum     = 20*jnp.log10((abs(p_mL_H_abs + p_mT_H_abs))/p_ref) 
    res.SPL_prop_harmonic_bpf_spectrum_dBA = A_weighting(res.SPL_prop_harmonic_bpf_spectrum,res.f[:,:,:,0,:]) 
    res.SPL_prop_harmonic_1_3_spectrum     = convert_to_one_third_octave_band(res.SPL_prop_harmonic_bpf_spectrum,res.f[:,0,0,0,:],settings)
    res.SPL_prop_harmonic_1_3_spectrum_dBA = convert_to_one_third_octave_band(res.SPL_prop_harmonic_bpf_spectrum_dBA,res.f[:,0,0,0,:],settings)
    res.SPL_prop_harmonic_1_3_spectrum     = jnp.where(jnp.isinf(res.SPL_prop_harmonic_1_3_spectrum),0,res.SPL_prop_harmonic_1_3_spectrum)
    res.SPL_prop_harmonic_1_3_spectrum_dBA = jnp.where(jnp.isinf(res.SPL_prop_harmonic_1_3_spectrum_dBA),0,res.SPL_prop_harmonic_1_3_spectrum_dBA)

    return