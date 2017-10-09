## @ingroup Methods-Propulsion
# fm_id.py
# 
# Created:  ### ####, SUAVE Team
# Modified: Feb 2016, E. Botero
import SUAVE


import numpy as np

# ----------------------------------------------------------------------
#  fm_id
# ----------------------------------------------------------------------

## @ingroup Methods-Propulsion
def theta_beta_mach(M0, theta, gamma, delta):
    l1 = np.sqrt(((M0**2-1)**2)-3*((1+((gamma-1)/2)*M0**2)*(1+((gamma+1)/2)*M0**2))*(np.tan(theta))**2)
    l2 = ((M0**2-1)**3-9*(1+(gamma-1)/2*M0**2)*(1+(gamma-1)/2*M0**2+(gamma+1)/4*M0**4)*(np.tan(theta))**2)/(l1**3)
    beta = np.arctan((M0**2-1+2*l1*np.cos((4*np.pi*delta+np.arccos(l2))/3))/(3*(1+(gamma-1)/2*M0**2)*np.tan(theta)))
    
    return beta
    
def oblique_shock_relation(M0, gamma, theta, beta):
    M0_n  = M0*np.sin(beta)
    M1_n  = np.sqrt((1+(gamma-1)/2*M0_n**2)/(gamma*M0_n**2-(gamma-1)/2))
    M1    = M1_n/np.sin(beta-theta)
    P_r   = 1+(2*gamma/(gamma+1))*(M1_n**2-1)
    T_r   = P_r*(((gamma-1)*M1_n**2+2)/((gamma+1)*M1_n**2))
    Pt_r  = (((gamma+1)*(M0*np.sin(beta))**2)/((gamma-1)*(M0*np.sin(beta))**2+2))**(gamma/(gamma-1))*((gamma+1)/(2*gamma*(M0*np.sin(beta))**2-(gamma-1)))**(1/(gamma-1)) 
    
    return M1, T_r, P_r, Pt_r