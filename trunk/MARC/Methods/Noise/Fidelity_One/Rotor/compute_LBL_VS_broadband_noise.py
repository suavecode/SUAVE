## @ingroup Methods-Noise-Fidelity_One-Rotor
# compute_TIP_broadband_noise.py
#
# Created: Feb 2023, M. Clarke 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np   

## @ingroup Methods-Noise-Fidelity_One-Rotor
def compute_LBL_VS_broadband_noise(R_c,alpha_star,delta_p,r_e,L,M,Dbar_h,f,U):
    '''This computes the laminar boundary layer compoment of broadband noise using the method outlined by the 
    Brooks, Pope and Marcolini (BPM) Model
    
    Assumptions:
        BPM models assumes a naca 0012 airfol  
        
    Source:  
        BPM Model:  Brooks, Thomas F., D. Stuart Pope, and Michael A.
        Marcolini. Airfoil self-noise and prediction. No. L-16528. 1989.
    
    Inputs:  
       R_c          - Reynolds number                        [-]
       alpha_star   - angle of attack blade section          [deg]
       delta_p      - boundary layer of pressure section     [m
       L            - length of blade section                [m]
       U            - velocity of blade section              [m/s]
       M            - Mach number                            [-]
       c            - airfoil section chord                  [m] 
       f            - frequency spectrum                     [Hz]
       Dbar_h       - high frequency directivity term        [-]
       r_e          - distance from noise source to observer [m] 
    
    Outputs 
       SPL_LBL_VS   - Sound pressure level of laminar boundary layer [dB]
       
    Properties Used:
        N/A   
    '''       
    
    St_prime              = f*delta_p/U # eqn 54 
    St_prime_1            = 0.001756*(R_c**0.3931)  # eqn 55 
    St_prime_1[R_c<1.3E5] = 0.18                    # eqn 55 
    St_prime_1[R_c>4E5]   = 0.28                    # eqn 55 
    
    St_prime_peak = St_prime_1*(10**(-0.04*alpha_star))
    
    G_1   = compute_G_1(St_prime/St_prime_peak)
    R_c_0 = compute_R_c_0(alpha_star)
    G_2   = compute_G_2(R_c/R_c_0)
    G_3   = compute_G_3(alpha_star)
    
    SPL_LBL_VS =  10*np.log10((delta_p*(M**5)*L*Dbar_h)/(r_e**2) ) + G_1  + G_2 + G_3 # eqn 53

    return  SPL_LBL_VS

def compute_G_1(e): 
    '''This computes the G_1 function using the BPM model
    
    Assumptions:
        BPM models assumes a naca 0012 airfol 
        Corrections made to match experimental results 
        
    Source:  
        BPM Model:  Brooks, Thomas F., D. Stuart Pope, and Michael A.
        Marcolini. Airfoil self-noise and prediction. No. L-16528. 1989.
    
    Inputs: 
        e     [-]
    
    Outputs 
        G_1   [-]
       
    Properties Used:
        N/A   
    '''         
    e     = e * 0.3  
    num_1 = 39.8 *0.5 
    num_2 = 98.409 *0.5  
    
    G_1           = -num_1*np.log10(e) - 11.2   # eqn 57
    G_1[e<1.64]   = -num_2*np.log10(e[e<1.64]) + 2  # eqn 57
    G_1[e<1.17]   = -5.076 + np.sqrt( 2.484 - 506.25*(np.log10(e[e<1.17]))**2)  # eqn 57
    G_1[e<0.8545] = num_2*np.log10(e[e<0.8545]) + 2      # eqn 57
    G_1[e<0.5974] = num_1*np.log10(e[e<0.5974]) - 11.2    # eqn 57
    
    return G_1

def compute_G_2(d):
    '''This computes the G_2 function using the BPM model
    
    Assumptions:
        BPM models assumes a naca 0012 airfol  
        
    Source:  
        BPM Model:  Brooks, Thomas F., D. Stuart Pope, and Michael A.
        Marcolini. Airfoil self-noise and prediction. No. L-16528. 1989.
    
    Inputs: 
        d     [-]
    
    Outputs 
        G_2   [-]
       
    Properties Used:
        N/A   
    '''     
 
    G_2           = -77.852*np.log10(d) + 15.328           # eqn 58 
    G_2[d<3.0889] = -65.188*np.log10(d[d<3.0889]) + 9.125  # eqn 58 
    G_2[d<1.7579] = -114.052*((np.log10(d[d<1.7579]))**2 ) # eqn 58
    G_2[d<0.5689] = 65.188*np.log10(d[d<0.5689]) + 9.125   # eqn 58
    G_2[d<0.3237] = 77.852*np.log10(d[d<0.3237]) + 15.328  # eqn 58    
    
    return G_2

def compute_G_3(alpha_star): 
    '''This computes the G_3 function using the BPM model
    
    Assumptions:
        BPM models assumes a naca 0012 airfol  
        
    Source:  
        BPM Model:  Brooks, Thomas F., D. Stuart Pope, and Michael A.
        Marcolini. Airfoil self-noise and prediction. No. L-16528. 1989.
    
    Inputs: 
        alpha_star    [deg]
    
    Outputs 
        G_3           [-]
       
    Properties Used:
        N/A   
    '''     
    G_3 = 171.04 - 3.03*alpha_star
    return G_3

def compute_R_c_0(alpha_star):
    '''This computes the R_c_0 function using the BPM model
    
    Assumptions:
        BPM models assumes a naca 0012 airfol  
        
    Source:  
        BPM Model:  Brooks, Thomas F., D. Stuart Pope, and Michael A.
        Marcolini. Airfoil self-noise and prediction. No. L-16528. 1989.
    
    Inputs: 
        alpha_star    [deg]
    
    Outputs 
        R_c_0         [-]
       
    Properties Used:
        N/A   
    '''     
    R_c_0               = 10**(0.215*alpha_star + 4.978) # eqn 59
    R_c_0[3<alpha_star] = 10**(0.120*alpha_star[3<alpha_star] + 5.263)  # eqn 59 
    return R_c_0 