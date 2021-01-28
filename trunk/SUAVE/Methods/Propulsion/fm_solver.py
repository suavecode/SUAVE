## @ingroup Methods-Propulsion
# fm_solver.py
#
# Created:  Sep 2017, P Goncalves
# Modified: Jan 2018, W. Maier

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from scipy.optimize import fsolve

# ----------------------------------------------------------------------
#  fm_solver
# ----------------------------------------------------------------------

## @ingroup Methods-Propulsion


def fm_solver(area_ratio, M0, gamma):
    """Function that takes in an area ratio and a Mach number associated to
    one of the areas and outputs the missing Mach number.

    Inputs:
    M           [-]
    gamma       [-]
    area_ratio  [-]

    Outputs:
    M1          [-]

    Source:
    https://web.stanford.edu/~cantwell/AA210A_Course_Material/AA210A_Course_Notes/
    """
    # Area-Mach Function
    func = lambda M1: ((M0/M1*((1.+(gamma-1.)/2.*M1*M1)/(1.+(gamma-1.)/2.*M0*M0))**((gamma+1.)/(2.*(gamma-1.))))-area_ratio)

    # Initializing the array
    M1_guess = np.ones_like(M0)

    # Separating supersonic and subsonic solutions
    i_low = M0 < 1.0
    i_high = M0 >= 1.0

    # Subsonic solution initialization
    M1_guess[i_low]= 0.1

    # Supersonic solution initialization
    M1_guess[i_high]= 1.1

    # Solving
    M1 = fsolve(func,M1_guess, factor=0.1)

    return M1
