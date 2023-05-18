## @ingroup Methods-Noise-Fidelity_Zero-Rotor
# compute_TIP_broadband_noise.py
#
# Created: Feb 2023, M. Clarke 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np   

## @ingroup Methods-Noise-Fidelity_Zero-Rotor
def compute_TIP_broadband_noise(alpha_tip,M,c,c_0,f,Dbar_h,r_e):
    '''This computes the tip noise compoment of broadband noise using the method outlined by the 
    Brooks, Pope and Marcolini (BPM) Model
    
    Assumptions:
        BPM models assumes a naca 0012 airfol  
        
    Source:  
        BPM Model:  Brooks, Thomas F., D. Stuart Pope, and Michael A.
        Marcolini. Airfoil self-noise and prediction. No. L-16528. 1989.
    
    Inputs:  
       alpha_TIP - angle of attack of tip section         [deg]
       M         - Mach number                            [-]
       c         - airfoil section chord                  [m]
       c_0       - speed of sound                         [m/s]
       f         - frequency spectrum                     [Hz]
       Dbar_h    - high frequency directivity term        [-]
       r_e       - distance from noise source to observer [m] 
    
    Outputs 
       SPL_TIP   - Sound pressure level of tip            [dB]
       
    Properties Used:
        N/A   
    '''       
    l_div_c                = 0.023 + 0.0169*alpha_tip              # eqn 67 BPM Paper 
    l_div_c[2<alpha_tip]   = 0.0378 + 0.0095*alpha_tip[2<alpha_tip]
    l                      = l_div_c*c                    # eqn 63 BPM Paper 
    M_max_div_M            = (1 + 0.036*alpha_tip)        # eqn 64 BPM Paper
    M_max                  = M_max_div_M * M              # eqn 64 BPM Paper
    U_max                  = c_0*M_max 
    St_prime_prime         = f*l/U_max                    # eqn 62 BPM Paper 
    SPL_TIP                = 10*np.log10(((M**2)*(M_max**3)*(l**2)*Dbar_h)/(r_e**2)) -\
                             30.5*(np.log10(St_prime_prime + 0.3))**2 + 126 # eqn 61  
    return SPL_TIP