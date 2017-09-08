## @ingroup Methods-Propulsion
# fm_id.py
# 
# Created:  ### ####, SUAVE Team
# Modified: Feb 2016, E. Botero

import numpy as np

from scipy.optimize import fsolve

# ----------------------------------------------------------------------
#  fm_id
# ----------------------------------------------------------------------

## @ingroup Methods-Propulsion


def fm_solver(Aratio, M0, gamma):
    """
    Function that takes in an area ratio and a Mach number associated to
    one of the areas and outputs the missing Mach number
    
    Inputs:
    M       [dimensionless]
    gamma   [dimensionless]
    Aratio  [dimensionless]
    
    Outputs:
    M1      [dimensionless]
    
    """
    
    func = lambda M1: (M0/M1*((1+(gamma-1)/2*M1**2)/(1+(gamma-1)/2*M0**2))**((gamma+1)/(2*(gamma-1))))-Aratio

    i_low = M0 <= 1.0
    i_high = M0 > 1.0
    M1_guess = 1.0*M0/M0
    
    M1_guess[i_low]= .1
    M1_guess[i_high]= 1.1
 
    M1 = fsolve(func,M1_guess)
    
    return M1
