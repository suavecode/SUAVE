## @ingroup Methods-Noise-Fidelity_One-Rotor
# compute_noise_directivities.py
#
# Created: Feb 2023, M. Clarke 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np   

## @ingroup Methods-Noise-Fidelity_One-Rotor
def compute_noise_directivities(Theta_e,Phi_e,M):
    '''This computes the laminar boundary layer compoment of broadband noise using the method outlined by the 
    Brooks, Pope and Marcolini (BPM) Model
    
    Assumptions:
        BPM models assumes a naca 0012 airfol  
        
    Source:  
        BPM Model:  Brooks, Thomas F., D. Stuart Pope, and Michael A.
        Marcolini. Airfoil self-noise and prediction. No. L-16528. 1989.
    
    Inputs:  
       Theta_e      - Radiation angle with respect to free stream x (chordwise) [deg]
       Phi_e        - Radiation angle with respect to free stream y (spanwise)  [deg]
       M_c          - Convection Mach number                                    [-]  
       M            - Mach number                                               [-] 
    
    Outputs 
       Dbar_h       - high frequency directivity term                           [-]
       Dbar_l       - low frequency directivity term                            [-] 
       
    Properties Used:
       N/A   
    '''     
    M_c      = 0.8*M
    Dbar_h   = (2*np.sin((Theta_e/2))*(np.sin(Phi_e)**2))/((1+M*np.cos(Theta_e))*(1 + (M - M_c)*np.cos(Theta_e))**2) # eqn B1 
    Dbar_l   = ((np.sin(Theta_e)**2)*(np.sin(Phi_e)**2))/((1 + M*np.cos(Theta_e))**4)                                # eqn B2
    
    return Dbar_h,Dbar_l 