## @ingroup Methods-Propulsion
# fm_id.py
# 
# Created:  ### ####, SUAVE Team
# Modified: Feb 2016, E. Botero
import SUAVE

from scipy.optimize import fsolve
import numpy as np

from SUAVE.Methods.Propulsion.oblique_shock import theta_beta_mach, oblique_shock_relation

# ----------------------------------------------------------------------
#  fm_id
# ----------------------------------------------------------------------

## @ingroup Methods-Propulsion  
    
def inlet_conditions(M_entry,gamma, nbr_oblique_shock,theta):
    """
    Function that takes the Mach, gamma, number of expected oblique shocks
    and wedge angle of the inlet. It outputs the flow properties after interacting
    with the inlet ; it verifies if all oblique shocks actually take place depending
    on entry conditions
    
    Inputs:
    M_entry                 [dimensionless]
    gamma                   [dimensionless]
    nbr_oblique_shock       [dimensionless]
    theta                   [rad]
    
    
    Outputs:
    T_ratio                 [dimensionless]
    Pt_ratio                [dimensionless]
   
    
    """
    shock = 0
    T_ratio = 1.
    P_ratio = 1.
    Pt_ratio = 1.
    
    while shock < nbr_oblique_shock :
        #-- Enter n-th oblique shock
            
        beta = theta_beta_mach(M_entry, theta, gamma, 1)
        M1, Tr, Pr, Ptr = oblique_shock_relation(M_entry,gamma, theta, beta)       
        T_ratio = T_ratio*(1/Tr)
        P_ratio = P_ratio*(1/Pr)
        Pt_ratio = Pt_ratio*Ptr
        M_entry = M1
        shock = shock +1
        
        if np.any(M1 <= 1.0) :
            break
    
    return T_ratio, Pt_ratio