""" stagnation_temperature.py: compute stagnation temperature for a 1D ideal gas """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from f_of_M import f_of_M

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def stagnation_temperature(g,T,M):

    """  Tt = stagnation_temperature(g,T,M): compute stagnation temperature for a 1D ideal gas
    
         Inputs:    g = ratio of specific heats     (required)  (float)
                    T = absolute temperature        (required)  (floats)    
                    M = Mach number                 (required)  (floats)

         Outputs:   Tt = stagnation temperature                 (floats)

    """

    return T*f_of_M(g,M)