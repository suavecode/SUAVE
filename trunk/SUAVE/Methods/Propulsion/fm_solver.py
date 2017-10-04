## @ingroup Methods-Propulsion
# fm_solver.py
# 
# Created:  Sep 2017, P Goncalves

import numpy as np

from scipy.optimize import fsolve

# ----------------------------------------------------------------------
#  fm_solver
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

    #Initializing the array
    M1_guess = 1.0*M0/M0
    
    # Separating supersonic and subsonic solutions
    i_low = M0 <= 1.0
    i_high = M0 > 1.0

    #--Subsonic solution
    M1_guess[i_low]= .1
    
    #--Supersonic solution
    M1_guess[i_high]= 1.1
 
    M1 = fsolve(func,M1_guess, factor=0.1)
    
    return M1
