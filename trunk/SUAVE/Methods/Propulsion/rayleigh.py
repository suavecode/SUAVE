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


def rayleigh(gamma, M0, TtR):
    
    func = lambda M1: ((1+gamma*M0**2)**2*M1**2*(1+(gamma-1)*M1**2/2))/((1+gamma*M1**2)**2*M0**2*(1+(gamma-1)/2*M0**2)) - TtR[-1]

    if M0 > 1.0:
        M1_guess = 1.1
    else:
        M1_guess = .01
        
    M = fsolve(func,M1_guess)
    Ptr = (1+gamma*M0**2)/(1+gamma*M**2)*((1+(gamma-1)/2*M**2)/(1+(gamma-1)/2*M0**2))**(gamma/(gamma-1))
    
    return M, Ptr