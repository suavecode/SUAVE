## @ingroup Methods-Aerodynamics-Common-Gas_Dynamics
# Isentropic.py
#
# Created:  May 2019, M. Dethy
# Modified:  
#           

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from sympy import Symbol
from sympy.solvers import solve
# ----------------------------------------------------------------------
#  Isentropic Flow Relations
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Common-Gas_Dynamics
def isentropic_relations(M,gamma):
    """Computes isentropic flow quantites

    Assumptions:
    None

    Source:
    https://www.grc.nasa.gov/www/k-12/airplane/isentrop.html

    
    Inputs:
    Mach, M                              [-]
    Isentropic Expansion Factor, gamma   [-]

    Outputs:
    Temperature Ratio, T_o_Tt            [-]
    Pressure Ratio, P_o_Pt               [-]
    Density ratio, rho_o_rhot            [-]
    Area ratio, A_o_Astar                [-]
    Area-mach relation, f_m              [-]

    
    Properties Used:
    N/A
    """

    # Standard isentropic flow equations
    T_o_Tt       = (1 + (gamma - 1)/2 * M**2) ** (-1)
    P_o_Pt       = (1 + (gamma - 1)/2 * M**2) ** (-gamma/(gamma-1))
    rho_o_rhot   = (1 + (gamma - 1)/2 * M**2) ** (-1/(gamma-1))
    A_o_Astar    = 1/M * ((gamma+1)/2)**(-(gamma+1)/(2*(gamma-1))) * (1 + (gamma - 1)/2 * M**2) ** ((gamma+1)/(2*(gamma-1)))
    f_m          = 1/A_o_Astar

    return T_o_Tt, P_o_Pt, rho_o_rhot, A_o_Astar, f_m

def get_m(f_m_array, gamma_array, subsonic_flag):
    """The mach number from a given area-mach relation value

    Assumptions:
    None

    Source:
    Chapter 10 of:
    https://web.stanford.edu/~cantwell/AA210A_Course_Material/AA210A_Course_Notes/

    
    Inputs:
    Area-mach relation, f_m                             [-]
    Isentropic Expansion Factor, gamma                  [-]
    subsonic_flag (=1 if subsonic mach, = 0 otherwise)  [-]

    Outputs:
    Mach, M                                             [-]

    
    Properties Used:
    N/A
    """
    M_list = []
    if np.shape(f_m_array) != (0,):
        for i, f_m in enumerate(f_m_array):
            gamma = round(gamma_array[i], 2)
            f_m = round(f_m, 2)
            if f_m == 0.0:
                M_list.append(0.0)
            else:
                # Symbolically solve for mach number
                M = Symbol("M",real=True)
                A_o_Astar    = 1/M * ((gamma+1)/2)**(-(gamma+1)/(2*(gamma-1))) * (1 + (gamma - 1)/2 * M**2) ** ((gamma+1)/(2*(gamma-1)))
                M = np.array(solve(A_o_Astar - 1/f_m, M))
                M = abs(M)
                if subsonic_flag == 1:
                    M_reasonable = M[M > 0]
                    M_list.append(np.asscalar(M_reasonable[M_reasonable < 1]))
                else:
                    M_list.append(np.asscalar(M[M >= 1]))
        return M_list
    return
