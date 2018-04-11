## @ingroup Methods-Propulsion
# rayleighflow.py
#
# Created:  Dec 2017, W. Maier

import numpy as np
from scipy.optimize import fsolve

# ----------------------------------------------------------------------
#  rayleigh
# ----------------------------------------------------------------------

## @ingroup Methods-Propulsion
def rayleighflow(gamma, M0, TtR):
    """
    Function that takes in a input (output) Mach number and a stagnation
    temperature ratio and yields an output (input) Mach number, according
    to the Rayleigh line flow equation. The function also outputs the stagnation
    pressure ratio

    Inputs:
    M0      [dimensionless]
    gamma   [dimensionless]
    Ttr     [dimensionless]

    Outputs:
    M1      [dimensionless]
    Ptr     [dimensionless]

    """



func = lambda M1: ((1+gamma*M0**2)**2*M1**2*(1+(gamma-1)/2*M1**2))/((1+gamma*M1**2)**2*M0**2*(1+(gamma-1)/2*M0**2)) - TtR

    #Initializing the array
    M1_guess = 1.0*M0/M0

    # Separating supersonic and subsonic solutions
    i_low = M0 <= 1.0
    i_high = M0 > 1.0

    #--Subsonic solution
    M1_guess[i_low]= .01

    #--Supersonic solution
    M1_guess[i_high]= 1.1
    
    M1 = fsolve(func,M1_guess, factor=0.1)

    #Calculate stagnation pressure ratio
    Ptr = (1+gamma*M0**2)/(1+gamma*M1**2)*((1+(gamma-1)/2*M1**2)/(1+(gamma-1)/2*M0**2))**(gamma/(gamma-1))

    return M1, Ptr
