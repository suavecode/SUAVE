""" Gas_Dynamics.py: Methods for 1D compressible gasdynamics (ideal gas) """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def f_of_M(g,M):

    return 1 + ((g-1)/2)*M*M

def stagnation_temperature(gas,T,M,p=101325):

    return T*f_of_M(gas.compute_gamma(T),M)

def stagnation_pressure(gas,p,M,T=300):

    g = gas.compute_gamma(T)
    n = g/(g-1)

    return p*f_of_M(g,M)**n

def stagnation_density(gas,rho,M):

    raise NotImplementedError
          
def normal_shock(gas,M):

    raise NotImplementedError

def oblique_shock(gas,M):

    raise NotImplementedError

def area_ratio_from_mach(gas,M):

    raise NotImplementedError

def mach_from_area_ratio(gas,M):

    raise NotImplementedError

